import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

# Настройки страницы
st.set_page_config(
    page_title='HR Attrition Predictor',
    page_icon='▣',
    layout='wide'
)

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR  = os.path.join(BASE_DIR, 'data')

# Загрузка модели
@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(MODEL_DIR, 'attrition_model.pkl'))
    le_dept       = joblib.load(os.path.join(MODEL_DIR, 'le_department.pkl'))
    le_pos        = joblib.load(os.path.join(MODEL_DIR, 'le_position.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    X_train       = joblib.load(os.path.join(MODEL_DIR, 'X_train.pkl'))
    X_test        = joblib.load(os.path.join(MODEL_DIR, 'X_test.pkl'))
    y_test        = joblib.load(os.path.join(MODEL_DIR, 'y_test.pkl'))
    y_pred        = joblib.load(os.path.join(MODEL_DIR, 'y_pred.pkl'))
    y_prob        = joblib.load(os.path.join(MODEL_DIR, 'y_prob.pkl'))
    return model, le_dept, le_pos, feature_names, X_train, X_test, y_test, y_pred, y_prob

model, le_dept, le_pos, feature_names, X_train, X_test, y_test, y_pred, y_prob = load_model()


# Загрузка полной базы сотрудников + расчёт risk score для дашборда
@st.cache_data
def load_company_data():
    df_emp  = pd.read_csv(os.path.join(DATA_DIR, 'employees.csv'))
    df_skud = pd.read_csv(os.path.join(DATA_DIR, 'skud.csv'))

    # Те же агрегаты SKUD, что использовались при обучении модели
    skud_agg = df_skud.groupby('employee_id').agg(
        total_days     = ('present', 'count'),
        present_days   = ('present', 'sum'),
        late_days      = ('late', 'sum'),
        avg_work_hours = ('work_hours', 'mean')
    ).reset_index()
    skud_agg['attendance_rate'] = skud_agg['present_days'] / skud_agg['total_days']
    skud_agg['late_rate']       = skud_agg['late_days'] / skud_agg['total_days']

    df = df_emp.merge(skud_agg, on='employee_id')

    # Кодируем категории теми же энкодерами, что использовались при обучении
    df['department_enc'] = le_dept.transform(df['department'])
    df['position_enc']   = le_pos.transform(df['position'])

    # Собираем фичи в том порядке, который ожидает модель
    feat_df = pd.DataFrame({
        'age': df['age'],
        'department': df['department_enc'],
        'position': df['position_enc'],
        'salary': df['salary'],
        'got_raise': df['got_raise'],
        'raise_count': df['raise_count'],
        'months_since_last_raise': df['months_since_last_raise'],
        'has_training': df['has_training'],
        'trainings_count': df['trainings_count'],
        'months_since_last_training': df['months_since_last_training'],
        'performance': df['performance'],
        'enps': df['enps'],
        'avg_work_hours': df['avg_work_hours'],
        'attendance_rate': df['attendance_rate'],
        'late_rate': df['late_rate'],
    })[feature_names]

    df['risk_score'] = model.predict_proba(feat_df)[:, 1]
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Низкий', 'Средний', 'Высокий']
    )

    return df


# Справочники
DEPARTMENTS = ['Производство', 'Логистика', 'Финансы', 'HR', 'IT', 'Продажи']
POSITIONS = {
    'Производство': ['Оператор', 'Технолог', 'Мастер смены', 'Начальник цеха'],
    'Логистика':    ['Водитель', 'Кладовщик', 'Логист', 'Менеджер логистики'],
    'Финансы':      ['Бухгалтер', 'Экономист', 'Финансовый аналитик', 'CFO'],
    'HR':           ['HR-специалист', 'Рекрутер', 'HR BP', 'HR-директор'],
    'IT':           ['Разработчик', 'Аналитик', 'DevOps', 'IT-менеджер'],
    'Продажи':      ['Менеджер продаж', 'Старший менеджер', 'KAM', 'Директор продаж']
}

# Навигация — вертикальные вкладки в sidebar
st.sidebar.markdown('### Навигация')

if 'page' not in st.session_state:
    st.session_state.page = 'HR показатели'

if st.sidebar.button('Анализ сотрудника', width='stretch',
                      type='primary' if st.session_state.page == 'Анализ сотрудника' else 'secondary'):
    st.session_state.page = 'Анализ сотрудника'
    st.rerun()

if st.sidebar.button('Загрузка CSV', width='stretch',
                      type='primary' if st.session_state.page == 'Загрузка CSV' else 'secondary'):
    st.session_state.page = 'Загрузка CSV'
    st.rerun()

if st.sidebar.button('HR показатели', width='stretch',
                      type='primary' if st.session_state.page == 'HR показатели' else 'secondary'):
    st.session_state.page = 'HR показатели'
    st.rerun()

st.sidebar.markdown('---')

if st.sidebar.button('Метрики модели', width='stretch',
                      type='primary' if st.session_state.page == 'Метрики модели' else 'secondary'):
    st.session_state.page = 'Метрики модели'
    st.rerun()

page = st.session_state.page

# ─────────────────────────────────────────────
# СТРАНИЦА 0 — HR показатели
# ─────────────────────────────────────────────
if page == 'HR показатели':
    st.title('HR показатели')
    st.markdown('Обзор текущего состояния компании на основе данных HR и СКУД')

    df_company = load_company_data()

    # Берём только тех, кто сейчас работает (ещё не уволился)
    df_active = df_company[df_company['leave_date'].isna()].copy()

    # Средний стаж активных сотрудников в месяцах (на конец периода данных)
    reference_date = pd.to_datetime(df_company['hire_date']).max()  # последняя дата найма в базе как точка отсчёта
    hire_dates = pd.to_datetime(df_active['hire_date'])
    avg_tenure_months = ((reference_date - hire_dates).dt.days / 30.44).mean()

    # ── KPI-карточки ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Всего сотрудников', len(df_active))
    col2.metric('Средний стаж, мес.', f"{avg_tenure_months:.0f}")
    col3.metric('Средний eNPS', f"{df_active['enps'].mean():.0f}")
    col4.metric('Сотрудников в зоне риска',
                int((df_active['risk_level'] == 'Высокий').sum()))

    st.markdown('---')
    col_left, col_right = st.columns(2)

    # ── Распределение по уровню риска ──
    with col_left:
        st.subheader('Распределение по уровню риска')
        risk_counts = df_active['risk_level'].value_counts().reindex(
            ['Низкий', 'Средний', 'Высокий']
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_map = ['#2e8540', '#b8860b', '#c0392b']
        ax.bar(risk_counts.index, risk_counts.values, color=colors_map)
        ax.set_ylabel('Количество сотрудников')
        st.pyplot(fig)
        plt.close()

    # ── Средний риск по отделам (прогноз модели, текущая ситуация) ──
    with col_right:
        st.subheader('Средний риск по отделам (сейчас)')
        dept_risk = df_active.groupby('department')['risk_score'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_risk.index, dept_risk.values * 100, color='#c0392b')
        ax.set_ylabel('Средний риск, %')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    col_left2, col_right2 = st.columns(2)

    # ── Текучка по отделам (исторически, по всей базе) ──
    with col_left2:
        st.subheader('Текучка по отделам (исторически)')
        dept_attr = df_company.groupby('department')['attrition'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_attr.index, dept_attr.values * 100, color='steelblue')
        ax.set_ylabel('Доля уволившихся, %')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    # ── Численность отделов сейчас ──
    with col_right2:
        st.subheader('Численность отделов')
        dept_count = df_active['department'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_count.index, dept_count.values, color='#555555')
        ax.set_ylabel('Сотрудников')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # ── Топ-10 сотрудников по риску прямо сейчас ──
    st.subheader('Топ-10 сотрудников по риску увольнения')
    st.caption(
        f'Из них в зоне высокого риска (>60%): '
        f'{int((df_active["risk_level"] == "Высокий").sum())} человек'
    )
    top_risk = df_active.sort_values('risk_score', ascending=False).head(10)
    top_risk_display = top_risk[['name', 'department', 'position',
                                  'risk_score', 'risk_level', 'enps', 'performance']].copy()
    top_risk_display['risk_score'] = (top_risk_display['risk_score'] * 100).round(1)
    top_risk_display.columns = ['Сотрудник', 'Отдел', 'Должность',
                                 'Риск, %', 'Уровень', 'eNPS', 'Производительность']
    st.dataframe(top_risk_display.reset_index(drop=True), width='stretch')

# ─────────────────────────────────────────────
# СТРАНИЦА 1 — Анализ одного сотрудника
# ─────────────────────────────────────────────
elif page == 'Анализ сотрудника':
    st.title('HR Attrition Risk Predictor')
    st.markdown('Введите данные сотрудника чтобы получить оценку риска увольнения')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Данные сотрудника')
        age         = st.slider('Возраст', 18, 60, 30)
        department  = st.selectbox('Отдел', DEPARTMENTS)
        position    = st.selectbox('Должность', POSITIONS[department])
        salary      = st.slider('Зарплата (USD)', 200, 1500, 600)
        raise_count = st.slider('Сколько раз повышали зарплату', 0, 5, 1)

        if raise_count == 0:
            got_raise = 'Нет'
            st.caption('Повышений зарплаты не было')
            months_since_last_raise = st.slider(
                'Месяцев с начала работы без повышения', 0, 36, 12
            )
        else:
            got_raise = 'Да'
            months_since_last_raise = st.slider(
                'Месяцев с последнего повышения', 0, 36, 6
            )

        st.markdown('---')
        st.subheader('Обучение и развитие')
        trainings_count = st.slider('Сколько обучений/курсов пройдено', 0, 8, 1)

        if trainings_count == 0:
            has_training = 'Нет'
            st.caption('Обучений не было')
            months_since_last_training = st.slider(
                'Месяцев с начала работы без обучения', 0, 30, 12
            )
        else:
            has_training = 'Да'
            months_since_last_training = st.slider(
                'Месяцев с последнего обучения', 0, 30, 6
            )

        st.markdown('---')
        performance = st.slider('Оценка производительности', 1.0, 5.0, 3.5, 0.1)
        enps        = st.slider('eNPS сотрудника', -100, 100, 30)

    with col2:
        st.subheader('Данные СКУД')
        attendance_rate = st.slider('Посещаемость (%)', 50, 100, 95) / 100
        late_rate       = st.slider('Доля опозданий (%)', 0, 50, 10) / 100
        avg_work_hours  = st.slider('Среднее рабочих часов', 4.0, 12.0, 8.0, 0.1)

    st.markdown('---')

    if st.button('Рассчитать риск увольнения', width='stretch'):

        dept_encoded         = le_dept.transform([department])[0]
        pos_encoded          = le_pos.transform([position])[0]
        got_raise_encoded    = 1 if got_raise == 'Да' else 0
        has_training_encoded = 1 if has_training == 'Да' else 0

        values_map = {
            'age': age,
            'department': dept_encoded,
            'position': pos_encoded,
            'salary': salary,
            'got_raise': got_raise_encoded,
            'raise_count': raise_count,
            'months_since_last_raise': months_since_last_raise,
            'has_training': has_training_encoded,
            'trainings_count': trainings_count,
            'months_since_last_training': months_since_last_training,
            'performance': performance,
            'enps': enps,
            'avg_work_hours': avg_work_hours,
            'attendance_rate': attendance_rate,
            'late_rate': late_rate,
        }

        features = pd.DataFrame([[values_map[f] for f in feature_names]],
                                 columns=feature_names)

        prob = model.predict_proba(features)[0][1]

        col_res, col_rec = st.columns(2)

        with col_res:
            st.subheader('Результат')
            if prob < 0.3:
                color = '#2e8540'
                level = 'Низкий риск'
            elif prob < 0.6:
                color = '#b8860b'
                level = 'Средний риск'
            else:
                color = '#c0392b'
                level = 'Высокий риск'

            st.markdown(
                f"<h3 style='color:{color}'>● {level}</h3>",
                unsafe_allow_html=True
            )
            st.markdown(f"<h1 style='color:{color}'>{prob*100:.1f}%</h1>",
                        unsafe_allow_html=True)
            st.progress(float(prob))

        with col_rec:
            st.subheader('Рекомендация HR')
            if prob < 0.3:
                st.success('''
                **Наблюдение**
                - Сотрудник стабилен
                - Плановые встречи 1-on-1
                - Поддерживать текущие условия
                ''')
            elif prob < 0.6:
                st.warning('''
                **Требует внимания**
                - Провести беседу с руководителем
                - Проверить удовлетворённость работой
                - Рассмотреть возможность повышения или обучения
                - Назначить встречу с HR в течение 2 недель
                ''')
            else:
                st.error('''
                **Высокий приоритет**
                - Срочная беседа с HR и руководителем
                - Проверить уровень вовлечённости (eNPS)
                - Пересмотреть зарплату и нагрузку
                - Предложить программу обучения/развития
                - Выяснить причины низкой посещаемости
                - Разработать план удержания
                ''')

        st.markdown('---')
        st.subheader('Ключевые факторы риска')

        factors = []
        if performance < 2.5:
            factors.append('— Низкая производительность')
        if enps < 0:
            factors.append('— Отрицательный eNPS: сотрудник недоволен')
        if attendance_rate < 0.90:
            factors.append('— Низкая посещаемость')
        if not got_raise_encoded:
            factors.append('— Не было повышения зарплаты')
        if months_since_last_raise > 18:
            factors.append('— Давно не было повышения зарплаты (18+ месяцев)')
        if not has_training_encoded:
            factors.append('— Сотрудник не проходил обучение')
        if months_since_last_training > 15:
            factors.append('— Давно не было обучения (15+ месяцев)')
        if avg_work_hours < 6:
            factors.append('— Мало рабочих часов')

        if factors:
            for f in factors:
                st.markdown(f)
        else:
            st.markdown('— Явных факторов риска не обнаружено')

        st.markdown('---')
        st.subheader('Объяснение решения модели · SHAP')

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values    = shap_values[0],
                base_values = explainer.expected_value,
                data      = features.iloc[0],
                feature_names = feature_names
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

        st.caption('Красные факторы увеличивают риск увольнения · синие снижают')

# ─────────────────────────────────────────────
# СТРАНИЦА 2 — Метрики модели
# ─────────────────────────────────────────────
elif page == 'Метрики модели':
    st.title('Метрики качества модели')

    roc_auc = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred,
                target_names=['Остался', 'Уволился'], output_dict=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('ROC-AUC',   f"{roc_auc:.3f}")
    col2.metric('Accuracy',  f"{report['accuracy']:.3f}")
    col3.metric('Recall (уволился)',    f"{report['Уволился']['recall']:.3f}")
    col4.metric('F1 (уволился)',        f"{report['Уволился']['f1-score']:.3f}")

    st.markdown('---')
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('ROC-кривая')
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1], [0,1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-кривая')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader('Матрица ошибок')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Остался', 'Уволился'],
                    yticklabels=['Остался', 'Уволился'])
        ax.set_ylabel('Реально')
        ax.set_xlabel('Предсказано')
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader('Важность признаков')
        fi = pd.DataFrame({
            'feature':    feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(fi['feature'], fi['importance'], color='steelblue')
        ax.set_xlabel('Важность')
        st.pyplot(fig)
        plt.close()

    with col_right2:
        st.subheader('Распределение вероятностей риска')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6,
                color='steelblue', label='Остался')
        ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6,
                color='salmon', label='Уволился')
        ax.axvline(0.3, color='orange', linestyle='--', label='Порог 30%')
        ax.axvline(0.6, color='red',    linestyle='--', label='Порог 60%')
        ax.set_xlabel('Вероятность увольнения')
        ax.set_ylabel('Количество')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.subheader('Детальный отчёт')
    df_report = pd.DataFrame(report).transpose().round(3)
    st.dataframe(df_report, width='stretch')

# ─────────────────────────────────────────────
# СТРАНИЦА 3 — Загрузка CSV
# ─────────────────────────────────────────────
elif page == 'Загрузка CSV':
    st.title('Пакетный анализ сотрудников')
    st.markdown('Загрузите CSV файл со списком сотрудников для получения risk score')

    st.subheader('Шаблон CSV')
    template = pd.DataFrame([{
        'age': 30, 'department': 'IT', 'position': 'Аналитик',
        'salary': 800, 'got_raise': 1, 'raise_count': 1,
        'months_since_last_raise': 6,
        'has_training': 1, 'trainings_count': 2,
        'months_since_last_training': 5,
        'performance': 3.5, 'enps': 40, 'avg_work_hours': 8.0,
        'attendance_rate': 0.95, 'late_rate': 0.10
    }])
    st.dataframe(template)
    csv_template = template.to_csv(index=False).encode('utf-8-sig')
    st.download_button('Скачать шаблон', csv_template,
                       'template.csv', 'text/csv')

    st.markdown('---')

    uploaded = st.file_uploader('Загрузите CSV файл', type='csv')

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.subheader('Загруженные данные')
        st.dataframe(df_input, width='stretch')

        try:
            df_encoded = df_input.copy()
            df_encoded['department'] = le_dept.transform(df_input['department'])
            df_encoded['position']   = le_pos.transform(df_input['position'])

            X_batch = df_encoded[feature_names]
            probs   = model.predict_proba(X_batch)[:, 1]

            df_result = df_input.copy()
            df_result['risk_score_%'] = (probs * 100).round(1)
            df_result['risk_level'] = pd.cut(
                probs,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Низкий', 'Средний', 'Высокий']
            )
            df_result = df_result.sort_values('risk_score_%', ascending=False)

            st.markdown('---')
            st.subheader('Результаты')
            st.dataframe(df_result[['department', 'position', 'risk_score_%',
                                     'risk_level']].reset_index(drop=True),
                         width='stretch')

            csv_result = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button('Скачать результаты', csv_result,
                               'attrition_results.csv', 'text/csv')

        except Exception as e:
            st.error(f'Ошибка: {e}. Проверьте формат CSV по шаблону.')