"""Запуск ML-эксперимента + расширенные клинические фильтры.

Возможности:
- ~40 бинарных фильтров (ignore / только с / только без) по 9 категориям.
- ~12 числовых фильтров (min/max): возраст, ИМТ, AHI, ODI, SpO2,
  TST, эффективность сна, шкалы ESS/HADS, HbA1c, холестерин.
- Enum-фильтры: качество записи, курение, алкоголь.
- 4 пресета: «Все», «Чистая выборка», «Только контроль», «Только тяжёлые».
- Защита от перехода (state.begin_busy/end_busy) + восстановление лога.
- Автоподстановка combos (t_pos, buffer_sec) из кэша.
- Best-by-CV в сравнении нормализаций.
- Keep-alive для WebSocket.
"""
import asyncio
import traceback
from pathlib import Path

import pandas as pd
from nicegui import ui

from ui.state import state
from ui.components import help_expansion, section_header, warning_banner
from ui.logger import SafeLog, safe_notify, safe_set
from ui.help_content import HELP_EXPERIMENTS, HELP_CLINICAL_FILTERS
from ml.pipeline import (
    ExperimentConfig,
    run_experiment,
    run_comparison,
)
from ml.report import generate_report


# ------------------------------------------------------------------ поля
# Колонки, которые нужно приклеить из state.clinical (patient_clinical_data)
CLIN_FIELDS = [
    # Антропометрия
    "bmi", "height_cm", "weight_kg", "neck_circumference_cm",
    # ЛОР
    "ent_uvula", "ent_palate", "ent_tonsils",
    "ent_septum_deviation", "ent_micrognathia",
    # ССС
    "cvd_hypertension", "cvd_ihd", "cvd_heart_failure", "cvd_rhythm",
    "cvd_stroke", "cvd_aneurysm",
    # Эндокринные
    "endocrine_diabetes", "endocrine_prediabetes",
    "endocrine_hypothyreosis", "endocrine_hyperthyreosis",
    "endocrine_thyroid_nodes", "endocrine_menopause",
    "endocrine_testosterone_deficit",
    # Респираторные
    "resp_copd", "resp_asthma", "resp_allergic_rhinitis",
    # Неврология
    "neuro_epilepsy", "neuro_tbi", "neuro_dementia", "neuro_parkinson",
    "neuro_anxiety", "neuro_depression", "neuro_panic_attacks",
    # Сон
    "som_insomnia", "som_bruxism", "som_rls", "som_plmd",
    # ЖКТ
    "gi_gerd", "gi_gastritis", "gi_ulcer",
    "gi_cholecystitis", "gi_pancreatitis", "gi_hepatosis",
    # Онкология / урология / нефро
    "oncology_cancer", "uro_bph", "uro_nicturia", "nephro_ckd",
    # Операции / травмы
    "surgery_ent", "surgery_cardiac", "injury_trauma",
    # Шкалы
    "scale_ess", "scale_hads_anxiety", "scale_hads_depression",
    "scale_cat", "scale_insomnia_self",
    # Лаборатория
    "lab_tsh", "lab_glucose", "lab_hba1c",
    "lab_total_cholesterol", "lab_ldl", "lab_hdl", "lab_triglycerides",
    # Лекарства
    "meds_beta_blockers", "meds_benzodiazepines", "meds_antidepressants",
    "meds_antipsychotics", "meds_antiepileptics", "meds_hypnotics",
    "meds_melatonin", "meds_thyroxine", "meds_statins",
    # Образ жизни
    "smoking", "alcohol", "smoking_pack_years",
]


def _parse_available(available) -> list:
    parsed = []
    for item in available:
        try:
            if len(item) == 2:
                sid, l = item
                parsed.append((int(sid), float(l), 0.5, 30.0))
            elif len(item) >= 4:
                sid, l, t, b = item[:4]
                parsed.append((int(sid), float(l), float(t), float(b)))
        except (ValueError, TypeError):
            continue
    return parsed


# ------------------------------------------------------------------ merge
async def _load_and_merge(epoch_len: float,
                          t_pos: float = 0.5,
                          buffer_sec: float = 30.0) -> pd.DataFrame:
    """Признаки + метаданные + клиника (все нужные колонки)."""
    if state.api is None:
        raise ValueError(
            "API не настроен. Перейдите в ⚙ Settings, введите токен "
            "и нажмите «Сохранить»."
        )

    try:
        df = state.feature_store.load_all(
            epoch_len, t_pos=t_pos, buffer_sec=buffer_sec)
    except TypeError:
        df = state.feature_store.load_all(epoch_len)

    if df.empty:
        raise ValueError(
            f"Нет извлечённых данных для длины {epoch_len} с "
            f"(t_pos={t_pos:.2f}, buffer={buffer_sec:.0f} с). "
            f"Сначала выполните извлечение на шаге 2."
        )

    # Метаданные исследований
    if not state.studies:
        state.studies = await asyncio.to_thread(state.api.get_studies)
    if not state.clinical:
        state.clinical = await asyncio.to_thread(state.api.get_clinical)
    if not state.patients:
        state.patients = await asyncio.to_thread(state.api.get_patients)

    studies_df = pd.DataFrame(state.studies)
    study_keep = [c for c in [
        "study_id", "study_date", "rec_quality", "ahi", "odi",
        "total_sleep_time", "sleep_efficiency", "min_spo2", "avg_spo2",
        "breathing_impairment_severity",
    ] if c in studies_df.columns]
    if "sleep_data_quality" in studies_df.columns:
        studies_df = studies_df.rename(
            columns={"sleep_data_quality": "data_quality"})
        study_keep.append("data_quality")
    st = studies_df[study_keep].drop_duplicates("study_id")
    df = df.merge(st, on="study_id", how="left")

    # Возраст и пол — из patients + study_date
    if state.patients and "study_date" in df.columns:
        pats = pd.DataFrame(state.patients)
        if "birth_date" in pats.columns and "gender" in pats.columns:
            pats = pats[["patient_id", "birth_date", "gender"]].drop_duplicates(
                "patient_id")
            df = df.merge(pats, on="patient_id", how="left")
            try:
                df["birth_date"] = pd.to_datetime(
                    df["birth_date"], errors="coerce")
                df["study_date"] = pd.to_datetime(
                    df["study_date"], errors="coerce")
                days = (df["study_date"] - df["birth_date"]).dt.days
                df["age_at_study"] = (days / 365.25).round()
            except Exception:
                pass
            df["gender_code"] = (df["gender"] == "M").astype(int)

    # Клиника — много колонок
    if state.clinical:
        clin = pd.DataFrame(state.clinical).drop_duplicates(
            "patient_id", keep="last")
        clin_keep = [c for c in CLIN_FIELDS if c in clin.columns]
        clin = clin[["patient_id"] + clin_keep]
        # Убираем дубли, если такая колонка уже есть из studies
        to_drop = [c for c in clin_keep if c in df.columns and c not in ("bmi",)]
        if to_drop:
            clin = clin.drop(columns=[c for c in to_drop if c in clin.columns])
            clin_keep = [c for c in clin_keep if c in clin.columns]
        if clin_keep:
            df = df.merge(clin[["patient_id"] + clin_keep],
                          on="patient_id", how="left")
    else:
        for c in CLIN_FIELDS:
            if c not in df.columns:
                df[c] = None

    # Числовые колонки — конвертируем в числа
    for c in ["bmi", "age_at_study", "scale_ess", "scale_hads_anxiety",
              "scale_hads_depression", "lab_hba1c", "lab_total_cholesterol",
              "lab_tsh", "ahi", "odi", "min_spo2", "avg_spo2",
              "total_sleep_time", "sleep_efficiency"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ------------------------------------------------------------------ UI
def page_experiments():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "3. ML-эксперимент",
            "Обучение моделей + расширенные клинические фильтры",
        )

        help_expansion(
            "📖 Инструкция: параметры, минимум данных, план сравнения",
            HELP_EXPERIMENTS, icon="help",
        )
        help_expansion(
            "🩺 Клинические фильтры: что и зачем фильтровать",
            HELP_CLINICAL_FILTERS, icon="medical_services",
        )

        if state.api is None:
            warning_banner(
                "API не настроен. Перейдите в ⚙ Settings, введите токен "
                "и нажмите «Сохранить».",
                level="error")
            return

        # --- доступные данные ---
        available_raw = state.feature_store.list_available()
        available = _parse_available(available_raw)
        epoch_lengths = sorted({l for _, l, _, _ in available}) or [10]
        n_studies_cached = len({sid for sid, _, _, _ in available})

        if n_studies_cached == 0:
            warning_banner(
                "Нет извлечённых признаков. Сначала шаги 1 и 2.",
                level="warning")
        elif n_studies_cached < 10:
            warning_banner(
                f"⚠ ТОЛЬКО {n_studies_cached} пациентов. При split 80/20 "
                f"в test будет 1–2 пациента — статистически "
                f"бессмысленно. Нужно ≥20 (пилот), лучше 50+.",
                level="error")
        elif n_studies_cached < 20:
            warning_banner(
                f"⚠ {n_studies_cached} пациентов. Хватит для проверки "
                f"работоспособности, но НЕ для научного вывода.",
                level="warning")
        elif n_studies_cached < 50:
            warning_banner(
                f"В кэше {n_studies_cached} пациентов — пилотный уровень. "
                f"Для статьи желательно 50+.",
                level="info")

        # --- базовые параметры ---
        with ui.row().classes("gap-4 mb-2 items-center"):
            epoch_sel = ui.select(
                epoch_lengths, value=epoch_lengths[0],
                label="Длина эпохи (с)").classes("w-36")
            combo_sel = ui.select(
                {}, value=None,
                label="Параметры разметки",
            ).classes("w-56")
            target_sel = ui.select(
                {"has_apnea": "Апноэ в эпохе (label)",
                 "ahi_ge_15": "Тяжесть AHI ≥ 15 (по пациенту)"},
                value="has_apnea", label="Целевая переменная",
            ).classes("w-72")
            norm_sel = ui.select(
                {"none": "Без нормализации",
                 "global": "Global z-score",
                 "patient": "Patient-z (рекомендуется)"},
                value="patient", label="Нормализация",
            ).classes("w-56")

        ui.label("Модели:").classes("mt-2 font-bold")
        with ui.row().classes("gap-4 mb-2"):
            model_cbs = {}
            for name in ("logreg", "rf", "xgb", "mlp"):
                model_cbs[name] = ui.checkbox(
                    name, value=name in ("logreg", "rf"))

        # ============================================================
        # КЛИНИЧЕСКИЕ ФИЛЬТРЫ
        # ============================================================
        # Словари для доступа к виджетам при применении пресетов
        numeric_widgets = {}    # col → (cb, lo_in, hi_in)
        binary_widgets = {}     # col → select
        enum_widgets = {}       # col → select

        def _numeric_filter(parent, col, label, lo, hi, step=1, default_lo=None, default_hi=None):
            row = ui.row().classes("gap-2 items-center mb-1")
            with row:
                cb = ui.checkbox(label).classes("w-56")
                lo_in = ui.number(value=default_lo if default_lo is not None else lo,
                                  min=lo, max=hi, step=step).classes("w-24")
                ui.label("—").classes("text-gray-500")
                hi_in = ui.number(value=default_hi if default_hi is not None else hi,
                                  min=lo, max=hi, step=step).classes("w-24")
            numeric_widgets[col] = (cb, lo_in, hi_in)

        def _binary_filter(parent, col, label):
            row = ui.row().classes("gap-2 items-center mb-1")
            with row:
                sel = ui.select(
                    {"ignore": "не фильтровать",
                     "only_negative": "только БЕЗ",
                     "only_positive": "только С"},
                    value="ignore", label=label,
                ).classes("w-full")
            binary_widgets[col] = sel

        def _enum_filter(parent, col, label, options: dict):
            row = ui.row().classes("gap-2 items-center mb-1")
            with row:
                sel = ui.select(
                    {"ignore": "не фильтровать", **options},
                    value="ignore", label=label,
                ).classes("w-full")
            enum_widgets[col] = sel

        with ui.expansion(
            "🩺 Клинические фильтры (для подгрупп и чистых выборок)",
            icon="medical_services",
        ).classes("w-full mb-2 bg-gray-50").style("max-height: 600px"):

            # === Пресеты ===
            ui.label("Быстрые пресеты:").classes("font-bold text-sm mt-2")
            with ui.row().classes("gap-2 mb-3 flex-wrap"):
                ui.button("Все (сброс)",
                          on_click=lambda: _apply_preset("all")
                          ).props("outline dense")
                ui.button("Чистая выборка",
                          on_click=lambda: _apply_preset("clean")
                          ).props("color=primary dense")
                ui.button("Только контроль (AHI<5)",
                          on_click=lambda: _apply_preset("control")
                          ).props("color=green dense")
                ui.button("Только тяжёлые (AHI≥30)",
                          on_click=lambda: _apply_preset("severe")
                          ).props("color=red dense")

            # === 1. Демография ===
            with ui.expansion("👤 Демография и антропометрия",
                              icon="person").classes("w-full"):
                _numeric_filter(None, "age_at_study", "Возраст (лет)",
                                18, 90, 1)
                _numeric_filter(None, "bmi", "ИМТ (кг/м²)",
                                15, 50, 0.5)
                # Пол — через enum
                _enum_filter(
                    None, "gender_code", "Пол",
                    {"1": "только мужчины", "0": "только женщины"},
                )

            # === 2. Тяжесть ОАС ===
            with ui.expansion("📊 Тяжесть ОАС и сатурация",
                              icon="bar_chart").classes("w-full"):
                _numeric_filter(None, "ahi", "AHI (событий/ч)", 0, 150, 1)
                _numeric_filter(None, "odi", "ODI (событий/ч)", 0, 150, 1)
                _numeric_filter(None, "min_spo2", "Min SpO₂ (%)",
                                40, 100, 1)
                _numeric_filter(None, "avg_spo2", "Avg SpO₂ (%)",
                                50, 100, 0.5)

            # === 3. Качество данных ===
            with ui.expansion("✅ Качество данных и сон",
                              icon="check_circle").classes("w-full"):
                _numeric_filter(None, "total_sleep_time",
                                "TST (мин)", 0, 700, 10)
                _numeric_filter(None, "sleep_efficiency",
                                "Эффективность сна (%)", 0, 100, 1)
                _enum_filter(
                    None, "rec_quality", "Качество записи",
                    {"excellent": "excellent",
                     "good": "good",
                     "fair": "fair"},
                )
                _enum_filter(
                    None, "data_quality", "Качество данных сна",
                    {"excellent": "excellent",
                     "good": "good",
                     "fair": "fair"},
                )

            # === 4. Неврология ===
            with ui.expansion(
                "🧠 Неврология (сильно влияет на ЭЭГ!)",
                icon="psychology",
            ).classes("w-full"):
                ui.label(
                    "⚠ Эпилепсия, деменция, Паркинсон, ЧМТ искажают ЭЭГ — "
                    "для чистого сигнала их стоит исключать."
                ).classes("text-xs text-orange-700 mb-2")
                _binary_filter(None, "neuro_epilepsy", "Эпилепсия")
                _binary_filter(None, "neuro_dementia", "Деменция")
                _binary_filter(None, "neuro_parkinson", "Паркинсонизм")
                _binary_filter(None, "neuro_tbi", "ЧМТ в анамнезе")
                _binary_filter(None, "neuro_depression", "Депрессия")
                _binary_filter(None, "neuro_anxiety", "Тревожное расстройство")
                _binary_filter(None, "neuro_panic_attacks",
                               "Панические атаки")

            # === 5. Психотропы (самое важное!) ===
            with ui.expansion(
                "💊 Психотропные препараты (КРИТИЧНО для ЭЭГ!)",
                icon="medication",
            ).classes("w-full"):
                ui.label(
                    "⚠ Психотропы напрямую меняют спектр ЭЭГ. "
                    "Если пациент их принимает — модель может выучить "
                    "«пациента на препаратах» вместо «апноэ». "
                    "Для чистого исследования ЭЭГ — исключайте."
                ).classes("text-xs text-red-700 mb-2")
                _binary_filter(None, "meds_benzodiazepines",
                               "Бензодиазепины")
                _binary_filter(None, "meds_antidepressants",
                               "Антидепрессанты")
                _binary_filter(None, "meds_antipsychotics",
                               "Нейролептики")
                _binary_filter(None, "meds_antiepileptics",
                               "Антиэпилептики")
                _binary_filter(None, "meds_hypnotics", "Снотворные")
                _binary_filter(None, "meds_beta_blockers",
                               "Бета-блокаторы (влияют на ЧСС)")

            # === 6. Кардиометаболические ===
            with ui.expansion(
                "❤️ Кардиометаболические",
                icon="favorite",
            ).classes("w-full"):
                _binary_filter(None, "cvd_hypertension", "Гипертензия")
                _binary_filter(None, "cvd_ihd", "ИБС")
                _binary_filter(None, "cvd_heart_failure", "ХСН")
                _binary_filter(None, "cvd_stroke", "Инсульт")
                _binary_filter(None, "endocrine_diabetes", "СД2")
                _numeric_filter(None, "lab_hba1c", "HbA1c (%)",
                                4, 15, 0.1)
                _numeric_filter(None, "lab_total_cholesterol",
                                "Холестерин (ммоль/л)", 2, 12, 0.1)

            # === 7. Респираторные ===
            with ui.expansion(
                "🫁 Респираторные",
                icon="air",
            ).classes("w-full"):
                _binary_filter(None, "resp_copd", "ХОБЛ")
                _binary_filter(None, "resp_asthma", "Астма")

            # === 8. Другие коморбидности ===
            with ui.expansion(
                "🩻 Другие коморбидности",
                icon="healing",
            ).classes("w-full"):
                _binary_filter(None, "som_insomnia", "Инсомния")
                _binary_filter(None, "som_rls",
                               "Синдром беспокойных ног")
                _binary_filter(None, "som_plmd",
                               "Периодические движения")
                _binary_filter(None, "nephro_ckd", "ХБП")
                _binary_filter(None, "oncology_cancer", "Онкология")

            # === 9. Шкалы ===
            with ui.expansion(
                "📋 Шкалы (субъективные оценки)",
                icon="assignment",
            ).classes("w-full"):
                _numeric_filter(None, "scale_ess",
                                "ESS (0–24, сонливость)", 0, 24, 1)
                _numeric_filter(None, "scale_hads_anxiety",
                                "HADS-тревога (0–21)", 0, 21, 1)
                _numeric_filter(None, "scale_hads_depression",
                                "HADS-депрессия (0–21)", 0, 21, 1)

            # === 10. Образ жизни ===
            with ui.expansion(
                "🚬 Образ жизни",
                icon="smoking_rooms",
            ).classes("w-full"):
                _enum_filter(
                    None, "smoking", "Курение",
                    {"never": "только никогда",
                     "past": "только в прошлом",
                     "current": "только сейчас"},
                )
                _enum_filter(
                    None, "alcohol", "Алкоголь",
                    {"none": "только нет",
                     "moderate": "только умеренно"},
                )

        # ============================================================
        # Пресеты — реализация
        # ============================================================
        def _reset_all():
            for col, (cb, lo_in, hi_in) in numeric_widgets.items():
                try:
                    cb.value = False
                except Exception:
                    pass
            for col, sel in binary_widgets.items():
                try:
                    sel.value = "ignore"
                except Exception:
                    pass
            for col, sel in enum_widgets.items():
                try:
                    sel.value = "ignore"
                except Exception:
                    pass

        def _apply_preset(name: str):
            _reset_all()
            if name == "all":
                safe_notify("Все фильтры сброшены", type="info")
                return
            if name in ("clean", "control", "severe"):
                # Возраст 18-70
                if "age_at_study" in numeric_widgets:
                    cb, lo, hi = numeric_widgets["age_at_study"]
                    cb.value = True
                    lo.value = 18
                    hi.value = 70
                # TST >= 180
                if "total_sleep_time" in numeric_widgets:
                    cb, lo, hi = numeric_widgets["total_sleep_time"]
                    cb.value = True
                    lo.value = 180
                    hi.value = 700
                # Качество записи good+
                if "rec_quality" in enum_widgets:
                    enum_widgets["rec_quality"].value = "good"
                # Исключить неврологию
                for col in ("neuro_epilepsy", "neuro_dementia",
                            "neuro_parkinson", "neuro_tbi"):
                    if col in binary_widgets:
                        binary_widgets[col].value = "only_negative"
                # Исключить психотропы
                for col in ("meds_benzodiazepines", "meds_antidepressants",
                            "meds_antipsychotics", "meds_antiepileptics",
                            "meds_hypnotics"):
                    if col in binary_widgets:
                        binary_widgets[col].value = "only_negative"

            if name == "control":
                if "ahi" in numeric_widgets:
                    cb, lo, hi = numeric_widgets["ahi"]
                    cb.value = True
                    lo.value = 0
                    hi.value = 5
            elif name == "severe":
                if "ahi" in numeric_widgets:
                    cb, lo, hi = numeric_widgets["ahi"]
                    cb.value = True
                    lo.value = 30
                    hi.value = 150

            safe_notify(f"Пресет применён: {name}", type="positive")

        # ============================================================
        # Кнопки управления
        # ============================================================
        with ui.row().classes("gap-3 mt-4 items-center"):
            run_btn = ui.button("▶ Запустить эксперимент", color="primary")
            compare_btn = ui.button(
                "🔬 Сравнить нормализации", color="secondary")

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = SafeLog(ui.log(max_lines=1000).classes(
            "w-full h-72 font-mono text-xs"))

        # --------------------------------------------------- combo
        combos_map: dict = {}

        def _refresh_combos():
            nonlocal combos_map
            try:
                epoch_len = float(epoch_sel.value)
            except (TypeError, ValueError):
                return
            combos = sorted({
                (t, b) for _, l, t, b in available if l == epoch_len
            })
            if not combos:
                combos = [(0.5, 30.0)]
            labels = [f"t_pos={t:.2f}, buffer={int(b)}s" for t, b in combos]
            combos_map = dict(zip(labels, combos))
            try:
                combo_sel.options = labels
                combo_sel.value = labels[0]
                combo_sel.update()
            except Exception:
                pass

        _refresh_combos()
        try:
            epoch_sel.on_value_change(lambda e: _refresh_combos())
        except Exception:
            pass

        # --------------------------------------------------- collect
        def _collect_common():
            epoch_len = float(epoch_sel.value)
            t_pos, buffer_sec = combos_map.get(
                combo_sel.value, (0.5, 30.0))
            models = [n for n, cb in model_cbs.items() if cb.value]

            # Собираем фильтры
            active_filters = {}

            # Бинарные
            for col, sel in binary_widgets.items():
                try:
                    if sel.value != "ignore":
                        active_filters[col] = sel.value
                except Exception:
                    pass

            # Числовые
            for col, (cb, lo_in, hi_in) in numeric_widgets.items():
                try:
                    if not cb.value:
                        continue
                    lo_v = lo_in.value
                    hi_v = hi_in.value
                    lo = float(lo_v) if lo_v not in (None, "") else None
                    hi = float(hi_v) if hi_v not in (None, "") else None
                    if lo is None and hi is None:
                        continue
                    active_filters[col] = (lo, hi)
                except Exception:
                    pass

            # Enum
            for col, sel in enum_widgets.items():
                try:
                    if sel.value == "ignore":
                        continue
                    v = sel.value
                    if col == "gender_code":
                        # {enum_in: ["1"]} → но gender_code числовой
                        active_filters[col] = int(v)
                    elif col == "rec_quality" and v in (
                            "excellent", "good", "fair"):
                        active_filters[col] = {"quality_min": v}
                    elif col in ("smoking", "alcohol"):
                        active_filters[col] = {"enum_in": [v]}
                    else:
                        # data_quality
                        active_filters[col] = {"enum_in": [v]}
                except Exception:
                    pass

            return epoch_len, t_pos, buffer_sec, models, active_filters

        def _disable_all_buttons():
            try:
                run_btn.disable()
                compare_btn.disable()
            except Exception:
                pass

        def _enable_all_buttons():
            try:
                run_btn.enable()
                compare_btn.enable()
            except Exception:
                pass

        # --------------------------------------------------- dialog
        def _show_results_dialog(results: dict, report_path: str):
            ok = [(m, r) for m, r in results["models"].items()
                  if "auc" in r and "error" not in r]
            err = [(m, r) for m, r in results["models"].items()
                   if "error" in r]
            diag = results.get("diagnostics", {})

            with ui.dialog() as dlg, ui.card().classes(
                    "min-w-[720px] max-w-[95vw]"):
                ui.label("Результаты эксперимента").classes(
                    "text-xl font-bold mb-2")

                warnings = diag.get("warnings", [])
                if warnings:
                    for w in warnings:
                        ui.label(w).classes(
                            "text-sm text-orange-700 bg-orange-50 p-2 "
                            "rounded mb-1 whitespace-pre-wrap")
                    ui.separator()

                if ok:
                    ui.label("Test AUC:").classes("font-bold mt-2")
                    for m, r in ok:
                        a = r["auc"]
                        cv = r.get("cv_auc_mean")
                        cv_str = (f" | CV={cv:.3f}"
                                  if cv is not None else " | CV=N/A")
                        color = ("text-green-700" if a["auc"] >= 0.85
                                 else "text-blue-700" if a["auc"] >= 0.7
                                 else "text-orange-700")
                        ui.label(
                            f"{m}: AUC={a['auc']:.3f} "
                            f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
                            f"{cv_str}"
                        ).classes(f"font-mono {color}")

                    best_m, best_r = max(
                        ok, key=lambda kv: kv[1]["auc"]["auc"])
                    per = best_r.get("per_patient_test", [])
                    if per:
                        with ui.expansion(
                            f"Подробно по test-пациентам ({best_m})",
                            icon="person",
                        ).classes("w-full mt-2"):
                            for p in per:
                                auc_v = (f"AUC={p['auc']:.3f}"
                                         if p["auc"] is not None
                                         else f"AUC=n/a ({p['note']})")
                                ui.label(
                                    f"pid={p['patient_id']} | "
                                    f"{p['n_positive']}/{p['n_epochs']} "
                                    f"positive "
                                    f"({p['prevalence']*100:.1f}%) "
                                    f"| {auc_v}"
                                ).classes("font-mono text-xs")

                    best_auc = best_r["auc"]["auc"]
                    best_ci_low = best_r["auc"]["ci_low"]
                    if best_auc >= 0.85 and best_ci_low >= 0.80:
                        ui.label(
                            "✅ AUC ≥ 0.85 и CI_low ≥ 0.80 — хорошее "
                            "разделение классов. Для клиники нужна "
                            "независимая валидация."
                        ).classes("text-green-700 font-bold mt-2 "
                                  "whitespace-pre-wrap")
                    elif best_auc >= 0.70:
                        ui.label(
                            "⚠ AUC 0.70–0.85 — сигнал есть, но для "
                            "клинического скрининга недостаточно."
                        ).classes("text-blue-700 mt-2 "
                                  "whitespace-pre-wrap")
                    else:
                        ui.label(
                            "⚠ AUC < 0.70 — сигнала слабый или нет. "
                            "Причины смотрите в диагностике."
                        ).classes("text-orange-700 mt-2 "
                                  "whitespace-pre-wrap")

                if err:
                    ui.label(f"⚠ {len(err)} моделей упали:").classes(
                        "text-orange-600 font-bold mt-2")
                    for m, r in err:
                        ui.label(f"{m}: {r['error'][:250]}").classes(
                            "font-mono text-xs text-red-700")

                with ui.row().classes("mt-3"):
                    report_name = Path(report_path).name
                    ui.link("📄 Открыть полный отчёт",
                            target=f"/report/{report_name}",
                            new_tab=True).classes("text-blue-600")
                    ui.button("OK", on_click=dlg.close)
            dlg.open()

        # --------------------------------------------------- run
        async def run():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="warning")
                return
            epoch_len, t_pos, buffer_sec, models, active_filters = \
                _collect_common()
            if not models:
                safe_notify("Выберите хотя бы одну модель",
                            type="warning")
                return

            _disable_all_buttons()
            progress.value = 0
            state.begin_busy("experiment", "загрузка данных")
            try:
                n_filters = len(active_filters)
                log.push(
                    f"Параметры: len={epoch_len}s, t_pos={t_pos:.2f}, "
                    f"buffer={buffer_sec:.0f}s, фильтров активно: {n_filters}"
                )
                if active_filters:
                    log.push(f"  Фильтры: {list(active_filters.keys())}")

                df = await _load_and_merge(epoch_len, t_pos, buffer_sec)
                log.push(f"После merge: {len(df)} строк, "
                         f"{df['patient_id'].nunique()} пациентов")

                cfg = ExperimentConfig(
                    name=(f"{target_sel.value}_len{int(epoch_len)}_"
                          f"t{t_pos:.2f}_{norm_sel.value}"),
                    epoch_len=epoch_len,
                    models=models,
                    normalization=norm_sel.value,
                    target=target_sel.value,
                    overlap_t=t_pos,
                    comorbidity_filters=active_filters,
                )
                state.set_busy_progress(0.1, "обучение моделей")
                try:
                    status.set_text("Обучение моделей...")
                except Exception:
                    pass

                results = await asyncio.to_thread(
                    run_experiment, df, cfg,
                    state.config.experiments_dir,
                    lambda msg: log.push(msg))

                state.set_busy_progress(0.9, "формирование отчёта")
                report_path = (state.config.experiments_dir /
                               f"{cfg.name}_report.html")
                generate_report(results, str(report_path))
                state.set_busy_progress(1.0, "готово")

                try:
                    _show_results_dialog(results, str(report_path))
                except Exception as e:
                    log.push(f"(диалог не показан: {e})")
                safe_notify("Готово", type="positive")
            except Exception as e:
                log.push(f"FATAL: {e}\n{traceback.format_exc()}")
                safe_notify(f"Ошибка: {e}", type="negative")
            finally:
                _enable_all_buttons()
                state.end_busy()
                try:
                    progress.value = 1
                except Exception:
                    pass

        # --------------------------------------------------- compare
        async def compare():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="warning")
                return
            epoch_len, t_pos, buffer_sec, models, active_filters = \
                _collect_common()
            if not models:
                safe_notify("Выберите хотя бы одну модель", type="warning")
                return

            _disable_all_buttons()
            progress.value = 0
            state.begin_busy("comparison", "сравнение нормализаций")
            try:
                df = await _load_and_merge(epoch_len, t_pos, buffer_sec)
                log.push(f"После merge: {len(df)} строк, "
                         f"{df['patient_id'].nunique()} пациентов")

                base = ExperimentConfig(
                    name=(f"COMPARE_base_len{int(epoch_len)}_"
                          f"t{t_pos:.2f}"),
                    epoch_len=epoch_len,
                    models=models,
                    normalization="patient",
                    target=target_sel.value,
                    overlap_t=t_pos,
                    comorbidity_filters=active_filters,
                )
                state.set_busy_progress(0.1, "3 прогона")
                try:
                    status.set_text("Сравнение нормализаций...")
                except Exception:
                    pass

                summary = await asyncio.to_thread(
                    run_comparison, df, base,
                    state.config.experiments_dir,
                    ("none", "global", "patient"),
                    lambda msg: log.push(msg))

                state.set_busy_progress(1.0, "готово")

                try:
                    with ui.dialog() as dlg, ui.card().classes(
                            "min-w-[700px] max-w-[95vw]"):
                        ui.label("Сравнение нормализаций").classes(
                            "text-xl font-bold mb-2")
                        ui.label(
                            "Один и тот же split. Выбирайте по CV AUC!"
                        ).classes("text-sm text-gray-600 mb-3")

                        comp = summary["comparison"]
                        for m in models:
                            ui.label(f"— {m} —").classes(
                                "font-bold mt-2")
                            best_norm, best_cv = None, -1.0
                            for norm in ("none", "global", "patient"):
                                r = (comp.get(norm, {})
                                     .get("models", {})
                                     .get(m, {}))
                                auc_v = r.get("auc")
                                err = r.get("error")
                                if err:
                                    ui.label(
                                        f"  {norm}: ERROR {err[:80]}"
                                    ).classes(
                                        "font-mono text-xs text-red-600")
                                    continue
                                if auc_v is None:
                                    ui.label(
                                        f"  {norm}: n/a"
                                    ).classes("font-mono text-xs")
                                    continue
                                cv = r.get("cv_auc_mean")
                                ci_low = r.get("ci_low")
                                ci_high = r.get("ci_high")
                                ci_str = (
                                    f"[{ci_low:.3f}, {ci_high:.3f}]"
                                    if (ci_low is not None
                                        and ci_high is not None) else ""
                                )
                                cv_str = (f" | CV={cv:.3f}"
                                          if cv is not None else "")
                                color = (
                                    "text-green-700" if auc_v >= 0.7
                                    else "text-orange-600"
                                    if auc_v < 0.55
                                    else "text-blue-700")
                                ui.label(
                                    f"  {norm}: AUC={auc_v:.3f} "
                                    f"{ci_str}{cv_str}"
                                ).classes(f"font-mono text-sm {color}")
                                if cv is not None and cv > best_cv:
                                    best_cv, best_norm = cv, norm
                            if best_norm:
                                ui.label(
                                    f"  → лучший по CV: {best_norm} "
                                    f"(CV={best_cv:.3f})"
                                ).classes(
                                    "text-xs italic text-green-700")
                        ui.label(
                            f"Сводка: {summary['summary_path']}"
                        ).classes("text-xs text-gray-500 mt-3")
                        ui.button("OK", on_click=dlg.close)
                    dlg.open()
                except Exception as e:
                    log.push(f"(диалог не показан: {e})")

                safe_notify("Сравнение завершено", type="positive")
            except Exception as e:
                log.push(f"FATAL: {e}\n{traceback.format_exc()}")
                safe_notify(f"Ошибка: {e}", type="negative")
            finally:
                _enable_all_buttons()
                state.end_busy()
                try:
                    progress.value = 1
                except Exception:
                    pass

        # --------------------------------------------------- sync
        def _sync_from_state():
            try:
                if state.is_busy and state.busy_op in (
                        "experiment", "comparison"):
                    progress.value = state.busy_progress
                    if state.busy_message:
                        status.set_text(f"⚠ {state.busy_message}")
                    run_btn.disable()
                    compare_btn.disable()
                elif not state.is_busy:
                    if not run_btn.enabled:
                        run_btn.enable()
                    if not compare_btn.enabled:
                        compare_btn.enable()
            except Exception:
                pass

        def _keep_alive():
            safe_set(status, "text", status.text)

        ui.timer(0.5, _sync_from_state)
        ui.timer(20.0, _keep_alive)

        # --- восстановление после возврата ---
        if state.is_busy and state.busy_op in ("experiment", "comparison"):
            log.restore_from_state()
            try:
                status.set_text(f"⚠ Идёт операция: {state.busy_message}")
                progress.value = state.busy_progress
                run_btn.disable()
                compare_btn.disable()
            except Exception:
                pass

        run_btn.on_click(run)
        compare_btn.on_click(compare)