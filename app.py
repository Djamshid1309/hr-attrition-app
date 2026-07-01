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


TEXTS = {
    'RU': {
        # Навигация
        'nav_title':       'Навигация',
        'nav_analyze':     'Анализ сотрудника',
        'nav_csv':         'Загрузка CSV',
        'nav_hr':          'HR показатели',
        'nav_metrics':     'Метрики модели',
        'lang_label':      'Язык / Language',

        # Страница — Анализ сотрудника
        'analyze_title':   'HR Attrition Risk Predictor',
        'analyze_subtitle':'Введите данные сотрудника чтобы получить оценку риска увольнения',
        'sec_employee':    'Данные сотрудника',
        'field_age':       'Возраст',
        'field_dept':      'Отдел',
        'field_pos':       'Должность',
        'field_salary':    'Зарплата (USD)',
        'field_raises':    'Сколько раз повышали зарплату',
        'field_no_raise':  'Повышений зарплаты не было',
        'field_months_raise_no': 'Месяцев с начала работы без повышения',
        'field_months_raise':    'Месяцев с последнего повышения',
        'sec_training':    'Обучение и развитие',
        'field_trainings': 'Сколько обучений/курсов пройдено',
        'field_no_train':  'Обучений не было',
        'field_months_train_no': 'Месяцев с начала работы без обучения',
        'field_months_train':    'Месяцев с последнего обучения',
        'field_perf':      'Оценка производительности',
        'field_enps':      'eNPS сотрудника',
        'sec_skud':        'Данные СКУД',
        'field_attend':    'Посещаемость (%)',
        'field_late':      'Доля опозданий (%)',
        'field_hours':     'Среднее рабочих часов',
        'btn_calc':        'Рассчитать риск увольнения',
        'res_title':       'Результат',
        'rec_title':       'Рекомендация HR',
        'level_low':       'Низкий риск',
        'level_mid':       'Средний риск',
        'level_high':      'Высокий риск',
        'rec_low':         '**Наблюдение**\n- Сотрудник стабилен\n- Плановые встречи 1-on-1\n- Поддерживать текущие условия',
        'rec_mid':         '**Требует внимания**\n- Провести беседу с руководителем\n- Проверить удовлетворённость работой\n- Рассмотреть возможность повышения или обучения\n- Назначить встречу с HR в течение 2 недель',
        'rec_high':        '**Высокий приоритет**\n- Срочная беседа с HR и руководителем\n- Проверить уровень вовлечённости (eNPS)\n- Пересмотреть зарплату и нагрузку\n- Предложить программу обучения/развития\n- Выяснить причины низкой посещаемости\n- Разработать план удержания',
        'factors_title':   'Ключевые факторы риска',
        'f_low_perf':      '— Низкая производительность',
        'f_neg_enps':      '— Отрицательный eNPS: сотрудник недоволен',
        'f_attend':        '— Низкая посещаемость',
        'f_no_raise':      '— Не было повышения зарплаты',
        'f_old_raise':     '— Давно не было повышения зарплаты (18+ месяцев)',
        'f_no_train':      '— Сотрудник не проходил обучение',
        'f_old_train':     '— Давно не было обучения (15+ месяцев)',
        'f_hours':         '— Мало рабочих часов',
        'f_none':          '— Явных факторов риска не обнаружено',
        'shap_title':      'Объяснение решения модели · SHAP',
        'shap_caption':    'Красные факторы увеличивают риск увольнения · синие снижают',

        # Страница — HR показатели
        'hr_title':        'HR показатели',
        'hr_subtitle':     'Обзор текущего состояния компании на основе данных HR и СКУД',
        'kpi_total':       'Всего сотрудников',
        'kpi_tenure':      'Средний стаж, мес.',
        'kpi_enps':        'Средний eNPS',
        'kpi_risk':        'Сотрудников в зоне риска',
        'chart_risk_dist': 'Распределение по уровню риска',
        'chart_risk_dept': 'Средний риск по отделам (сейчас)',
        'chart_attr_hist': 'Текучка по отделам (исторически)',
        'chart_headcount': 'Численность отделов',
        'y_employees':     'Количество сотрудников',
        'y_risk_pct':      'Средний риск, %',
        'y_attr_pct':      'Доля уволившихся, %',
        'y_headcount':     'Сотрудников',
        'top10_title':     'Топ-10 сотрудников по риску увольнения',
        'top10_caption':   'Из них в зоне высокого риска (>60%):',
        'top10_person':    'чел.',
        'col_name':        'Сотрудник',
        'col_dept':        'Отдел',
        'col_pos':         'Должность',
        'col_risk':        'Риск, %',
        'col_level':       'Уровень',
        'col_enps':        'eNPS',
        'col_perf':        'Производительность',
        'risk_low':        'Низкий',
        'risk_mid':        'Средний',
        'risk_high':       'Высокий',

        # Страница — Метрики
        'metrics_title':   'Метрики качества модели',
        'met_roc':         'ROC-AUC',
        'met_acc':         'Accuracy',
        'met_recall':      'Recall (уволился)',
        'met_f1':          'F1 (уволился)',
        'chart_roc':       'ROC-кривая',
        'chart_cm':        'Матрица ошибок',
        'chart_fi':        'Важность признаков',
        'chart_dist':      'Распределение вероятностей риска',
        'cm_stayed':       'Остался',
        'cm_left':         'Уволился',
        'cm_actual':       'Реально',
        'cm_predicted':    'Предсказано',
        'fi_xlabel':       'Важность',
        'dist_xlabel':     'Вероятность увольнения',
        'dist_ylabel':     'Количество',
        'dist_stayed':     'Остался',
        'dist_left':       'Уволился',
        'thr_30':          'Порог 30%',
        'thr_60':          'Порог 60%',
        'report_title':    'Детальный отчёт',

        # Страница — Загрузка CSV
        'csv_title':       'Пакетный анализ сотрудников',
        'csv_subtitle':    'Загрузите CSV файл со списком сотрудников для получения risk score',
        'csv_template':    'Шаблон CSV',
        'csv_download_t':  'Скачать шаблон',
        'csv_upload':      'Загрузите CSV файл',
        'csv_loaded':      'Загруженные данные',
        'csv_results':     'Результаты',
        'csv_download_r':  'Скачать результаты',
        'csv_error':       'Ошибка: {}. Проверьте формат CSV по шаблону.',
    },
    'EN': {
        # Navigation
        'nav_title':       'Navigation',
        'nav_analyze':     'Employee Analysis',
        'nav_csv':         'Upload CSV',
        'nav_hr':          'HR Dashboard',
        'nav_metrics':     'Model Metrics',
        'lang_label':      'Language / Язык',

        # Page — Employee Analysis
        'analyze_title':   'HR Attrition Risk Predictor',
        'analyze_subtitle':'Enter employee data to get attrition risk score',
        'sec_employee':    'Employee Data',
        'field_age':       'Age',
        'field_dept':      'Department',
        'field_pos':       'Position',
        'field_salary':    'Salary (USD)',
        'field_raises':    'Number of salary raises',
        'field_no_raise':  'No salary raises yet',
        'field_months_raise_no': 'Months since hiring without a raise',
        'field_months_raise':    'Months since last raise',
        'sec_training':    'Training & Development',
        'field_trainings': 'Number of completed trainings/courses',
        'field_no_train':  'No trainings yet',
        'field_months_train_no': 'Months since hiring without training',
        'field_months_train':    'Months since last training',
        'field_perf':      'Performance score',
        'field_enps':      'Employee eNPS',
        'sec_skud':        'Attendance Data (SKUD)',
        'field_attend':    'Attendance rate (%)',
        'field_late':      'Late arrival rate (%)',
        'field_hours':     'Average working hours',
        'btn_calc':        'Calculate attrition risk',
        'res_title':       'Result',
        'rec_title':       'HR Recommendation',
        'level_low':       'Low Risk',
        'level_mid':       'Medium Risk',
        'level_high':      'High Risk',
        'rec_low':         '**Monitor**\n- Employee is stable\n- Schedule regular 1-on-1 meetings\n- Maintain current conditions',
        'rec_mid':         '**Needs Attention**\n- Have a conversation with the manager\n- Check job satisfaction\n- Consider a raise or training opportunity\n- Schedule an HR meeting within 2 weeks',
        'rec_high':        '**High Priority**\n- Urgent meeting with HR and manager\n- Check engagement level (eNPS)\n- Review salary and workload\n- Offer a development/training program\n- Investigate low attendance reasons\n- Develop a retention plan',
        'factors_title':   'Key Risk Factors',
        'f_low_perf':      '— Low performance score',
        'f_neg_enps':      '— Negative eNPS: employee is dissatisfied',
        'f_attend':        '— Low attendance rate',
        'f_no_raise':      '— No salary raise received',
        'f_old_raise':     '— No salary raise in 18+ months',
        'f_no_train':      '— Employee has not completed any training',
        'f_old_train':     '— No training in 15+ months',
        'f_hours':         '— Low average working hours',
        'f_none':          '— No significant risk factors detected',
        'shap_title':      'Model Explanation · SHAP',
        'shap_caption':    'Red factors increase attrition risk · blue factors decrease it',

        # Page — HR Dashboard
        'hr_title':        'HR Dashboard',
        'hr_subtitle':     'Overview of current workforce status based on HR and attendance data',
        'kpi_total':       'Total employees',
        'kpi_tenure':      'Avg tenure, months',
        'kpi_enps':        'Avg eNPS',
        'kpi_risk':        'Employees at high risk',
        'chart_risk_dist': 'Risk level distribution',
        'chart_risk_dept': 'Average risk by department (current)',
        'chart_attr_hist': 'Historical attrition by department',
        'chart_headcount': 'Department headcount',
        'y_employees':     'Number of employees',
        'y_risk_pct':      'Average risk, %',
        'y_attr_pct':      'Attrition rate, %',
        'y_headcount':     'Employees',
        'top10_title':     'Top-10 employees by attrition risk',
        'top10_caption':   'Of which at high risk (>60%):',
        'top10_person':    'employees',
        'col_name':        'Employee',
        'col_dept':        'Department',
        'col_pos':         'Position',
        'col_risk':        'Risk, %',
        'col_level':       'Level',
        'col_enps':        'eNPS',
        'col_perf':        'Performance',
        'risk_low':        'Low',
        'risk_mid':        'Medium',
        'risk_high':       'High',

        # Page — Metrics
        'metrics_title':   'Model Quality Metrics',
        'met_roc':         'ROC-AUC',
        'met_acc':         'Accuracy',
        'met_recall':      'Recall (attrited)',
        'met_f1':          'F1 (attrited)',
        'chart_roc':       'ROC Curve',
        'chart_cm':        'Confusion Matrix',
        'chart_fi':        'Feature Importance',
        'chart_dist':      'Risk Probability Distribution',
        'cm_stayed':       'Stayed',
        'cm_left':         'Left',
        'cm_actual':       'Actual',
        'cm_predicted':    'Predicted',
        'fi_xlabel':       'Importance',
        'dist_xlabel':     'Attrition probability',
        'dist_ylabel':     'Count',
        'dist_stayed':     'Stayed',
        'dist_left':       'Left',
        'thr_30':          'Threshold 30%',
        'thr_60':          'Threshold 60%',
        'report_title':    'Detailed Report',

        # Page — Upload CSV
        'csv_title':       'Batch Employee Analysis',
        'csv_subtitle':    'Upload a CSV file with employee list to get risk scores',
        'csv_template':    'CSV Template',
        'csv_download_t':  'Download template',
        'csv_upload':      'Upload CSV file',
        'csv_loaded':      'Uploaded data',
        'csv_results':     'Results',
        'csv_download_r':  'Download results',
        'csv_error':       'Error: {}. Please check the CSV format using the template.',
    }
}

DEPARTMENTS = {
    'RU': ['Производство', 'Логистика', 'Финансы', 'HR', 'IT', 'Продажи'],
    'EN': ['Производство', 'Логистика', 'Финансы', 'HR', 'IT', 'Продажи'],
}
POSITIONS = {
    'Производство': ['Оператор', 'Технолог', 'Мастер смены', 'Начальник цеха'],
    'Логистика':    ['Водитель', 'Кладовщик', 'Логист', 'Менеджер логистики'],
    'Финансы':      ['Бухгалтер', 'Экономист', 'Финансовый аналитик', 'CFO'],
    'HR':           ['HR-специалист', 'Рекрутер', 'HR BP', 'HR-директор'],
    'IT':           ['Разработчик', 'Аналитик', 'DevOps', 'IT-менеджер'],
    'Продажи':      ['Менеджер продаж', 'Старший менеджер', 'KAM', 'Директор продаж']
}


st.set_page_config(
    page_title='HR Attrition Predictor',
    page_icon='🔮',
    layout='wide'
)

# Пути
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR  = os.path.join(BASE_DIR, 'data')


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

@st.cache_data
def load_company_data():
    df_emp  = pd.read_csv(os.path.join(DATA_DIR, 'employees.csv'))
    df_skud = pd.read_csv(os.path.join(DATA_DIR, 'skud.csv'))
    skud_agg = df_skud.groupby('employee_id').agg(
        total_days     = ('present', 'count'),
        present_days   = ('present', 'sum'),
        late_days      = ('late', 'sum'),
        avg_work_hours = ('work_hours', 'mean')
    ).reset_index()
    skud_agg['attendance_rate'] = skud_agg['present_days'] / skud_agg['total_days']
    skud_agg['late_rate']       = skud_agg['late_days'] / skud_agg['total_days']
    df = df_emp.merge(skud_agg, on='employee_id')
    df['department_enc'] = le_dept.transform(df['department'])
    df['position_enc']   = le_pos.transform(df['position'])
    feat_df = pd.DataFrame({
        'age': df['age'], 'department': df['department_enc'],
        'position': df['position_enc'], 'salary': df['salary'],
        'got_raise': df['got_raise'], 'raise_count': df['raise_count'],
        'months_since_last_raise': df['months_since_last_raise'],
        'has_training': df['has_training'], 'trainings_count': df['trainings_count'],
        'months_since_last_training': df['months_since_last_training'],
        'performance': df['performance'], 'enps': df['enps'],
        'avg_work_hours': df['avg_work_hours'],
        'attendance_rate': df['attendance_rate'], 'late_rate': df['late_rate'],
    })[feature_names]
    df['risk_score'] = model.predict_proba(feat_df)[:, 1]
    return df


if 'lang' not in st.session_state:
    st.session_state.lang = 'RU'
if 'page' not in st.session_state:
    st.session_state.page = 'hr'

lang = st.session_state.lang

# Переключатель языка — две кнопки рядом
lang_col1, lang_col2 = st.sidebar.columns(2)
if lang_col1.button('RU', width='stretch',
                     type='primary' if lang == 'RU' else 'secondary'):
    st.session_state.lang = 'RU'
    st.rerun()
if lang_col2.button('EN', width='stretch',
                     type='primary' if lang == 'EN' else 'secondary'):
    st.session_state.lang = 'EN'
    st.rerun()

st.sidebar.markdown('---')

T = TEXTS[lang]

st.sidebar.markdown(f'### {T["nav_title"]}')

if st.sidebar.button(T['nav_analyze'], width='stretch',
                      type='primary' if st.session_state.page == 'analyze' else 'secondary'):
    st.session_state.page = 'analyze'
    st.rerun()

if st.sidebar.button(T['nav_csv'], width='stretch',
                      type='primary' if st.session_state.page == 'csv' else 'secondary'):
    st.session_state.page = 'csv'
    st.rerun()

if st.sidebar.button(T['nav_hr'], width='stretch',
                      type='primary' if st.session_state.page == 'hr' else 'secondary'):
    st.session_state.page = 'hr'
    st.rerun()

st.sidebar.markdown('---')

if st.sidebar.button(T['nav_metrics'], width='stretch',
                      type='primary' if st.session_state.page == 'metrics' else 'secondary'):
    st.session_state.page = 'metrics'
    st.rerun()

page = st.session_state.page


if page == 'hr':
    st.title(T['hr_title'])
    st.markdown(T['hr_subtitle'])

    df_company = load_company_data()
    df_active  = df_company[df_company['leave_date'].isna()].copy()

    # risk_level считаем с учётом языка
    df_active['risk_level'] = pd.cut(
        df_active['risk_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=[T['risk_low'], T['risk_mid'], T['risk_high']]
    )

    reference_date    = pd.to_datetime(df_company['hire_date']).max()
    hire_dates        = pd.to_datetime(df_active['hire_date'])
    avg_tenure_months = ((reference_date - hire_dates).dt.days / 30.44).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T['kpi_total'],  len(df_active))
    col2.metric(T['kpi_tenure'], f"{avg_tenure_months:.0f}")
    col3.metric(T['kpi_enps'],   f"{df_active['enps'].mean():.0f}")
    col4.metric(T['kpi_risk'],   int((df_active['risk_level'] == T['risk_high']).sum()))

    st.markdown('---')
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(T['chart_risk_dist'])
        risk_counts = df_active['risk_level'].value_counts().reindex(
            [T['risk_low'], T['risk_mid'], T['risk_high']]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(risk_counts.index, risk_counts.values,
               color=['#2e8540', '#b8860b', '#c0392b'])
        ax.set_ylabel(T['y_employees'])
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader(T['chart_risk_dept'])
        dept_risk = df_active.groupby('department')['risk_score'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_risk.index, dept_risk.values * 100, color='#c0392b')
        ax.set_ylabel(T['y_risk_pct'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader(T['chart_attr_hist'])
        dept_attr = df_company.groupby('department')['attrition'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_attr.index, dept_attr.values * 100, color='steelblue')
        ax.set_ylabel(T['y_attr_pct'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    with col_right2:
        st.subheader(T['chart_headcount'])
        dept_count = df_active['department'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dept_count.index, dept_count.values, color='#555555')
        ax.set_ylabel(T['y_headcount'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.subheader(T['top10_title'])
    high_risk_count = int((df_active['risk_level'] == T['risk_high']).sum())
    st.caption(f"{T['top10_caption']} {high_risk_count} {T['top10_person']}")

    top_risk = df_active.sort_values('risk_score', ascending=False).head(10)
    top_risk_display = top_risk[['name', 'department', 'position',
                                  'risk_score', 'risk_level', 'enps', 'performance']].copy()
    top_risk_display['risk_score'] = (top_risk_display['risk_score'] * 100).round(1)
    top_risk_display.columns = [T['col_name'], T['col_dept'], T['col_pos'],
                                 T['col_risk'], T['col_level'], T['col_enps'], T['col_perf']]
    st.dataframe(top_risk_display.reset_index(drop=True), width='stretch')


elif page == 'analyze':
    st.title(T['analyze_title'])
    st.markdown(T['analyze_subtitle'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(T['sec_employee'])
        age         = st.slider(T['field_age'], 18, 60, 30)
        department  = st.selectbox(T['field_dept'], DEPARTMENTS['RU'])
        position    = st.selectbox(T['field_pos'], POSITIONS[department])
        salary      = st.slider(T['field_salary'], 200, 1500, 600)
        raise_count = st.slider(T['field_raises'], 0, 5, 1)

        if raise_count == 0:
            got_raise = 0
            st.caption(T['field_no_raise'])
            months_since_last_raise = st.slider(T['field_months_raise_no'], 0, 36, 12)
        else:
            got_raise = 1
            months_since_last_raise = st.slider(T['field_months_raise'], 0, 36, 6)

        st.markdown('---')
        st.subheader(T['sec_training'])
        trainings_count = st.slider(T['field_trainings'], 0, 8, 1)

        if trainings_count == 0:
            has_training = 0
            st.caption(T['field_no_train'])
            months_since_last_training = st.slider(T['field_months_train_no'], 0, 30, 12)
        else:
            has_training = 1
            months_since_last_training = st.slider(T['field_months_train'], 0, 30, 6)

        st.markdown('---')
        performance = st.slider(T['field_perf'], 1.0, 5.0, 3.5, 0.1)
        enps        = st.slider(T['field_enps'], -100, 100, 30)

    with col2:
        st.subheader(T['sec_skud'])
        attendance_rate = st.slider(T['field_attend'], 50, 100, 95) / 100
        late_rate       = st.slider(T['field_late'], 0, 50, 10) / 100
        avg_work_hours  = st.slider(T['field_hours'], 4.0, 12.0, 8.0, 0.1)

    st.markdown('---')

    if st.button(T['btn_calc'], width='stretch'):
        dept_encoded = le_dept.transform([department])[0]
        pos_encoded  = le_pos.transform([position])[0]

        values_map = {
            'age': age, 'department': dept_encoded, 'position': pos_encoded,
            'salary': salary, 'got_raise': got_raise, 'raise_count': raise_count,
            'months_since_last_raise': months_since_last_raise,
            'has_training': has_training, 'trainings_count': trainings_count,
            'months_since_last_training': months_since_last_training,
            'performance': performance, 'enps': enps,
            'avg_work_hours': avg_work_hours,
            'attendance_rate': attendance_rate, 'late_rate': late_rate,
        }
        features = pd.DataFrame([[values_map[f] for f in feature_names]],
                                  columns=feature_names)
        prob = model.predict_proba(features)[0][1]

        col_res, col_rec = st.columns(2)

        with col_res:
            st.subheader(T['res_title'])
            if prob < 0.3:
                color = '#2e8540'
                level = T['level_low']
            elif prob < 0.6:
                color = '#b8860b'
                level = T['level_mid']
            else:
                color = '#c0392b'
                level = T['level_high']

            st.markdown(f"<h3 style='color:{color}'>● {level}</h3>",
                        unsafe_allow_html=True)
            st.markdown(f"<h1 style='color:{color}'>{prob*100:.1f}%</h1>",
                        unsafe_allow_html=True)
            st.progress(float(prob))

        with col_rec:
            st.subheader(T['rec_title'])
            if prob < 0.3:
                st.success(T['rec_low'])
            elif prob < 0.6:
                st.warning(T['rec_mid'])
            else:
                st.error(T['rec_high'])

        st.markdown('---')
        st.subheader(T['factors_title'])
        factors = []
        if performance < 2.5:            factors.append(T['f_low_perf'])
        if enps < 0:                     factors.append(T['f_neg_enps'])
        if attendance_rate < 0.90:       factors.append(T['f_attend'])
        if not got_raise:                factors.append(T['f_no_raise'])
        if months_since_last_raise > 18: factors.append(T['f_old_raise'])
        if not has_training:             factors.append(T['f_no_train'])
        if months_since_last_training > 15: factors.append(T['f_old_train'])
        if avg_work_hours < 6:           factors.append(T['f_hours'])

        if factors:
            for f in factors:
                st.markdown(f)
        else:
            st.markdown(T['f_none'])

        st.markdown('---')
        st.subheader(T['shap_title'])
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values      = shap_values[0],
                base_values = explainer.expected_value,
                data        = features.iloc[0],
                feature_names = feature_names
            ), show=False
        )
        st.pyplot(fig)
        plt.close()
        st.caption(T['shap_caption'])


elif page == 'metrics':
    st.title(T['metrics_title'])

    roc_auc = roc_auc_score(y_test, y_prob)
    report  = classification_report(
        y_test, y_pred,
        target_names=[T['cm_stayed'], T['cm_left']],
        output_dict=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T['met_roc'],    f"{roc_auc:.3f}")
    col2.metric(T['met_acc'],    f"{report['accuracy']:.3f}")
    col3.metric(T['met_recall'], f"{report[T['cm_left']]['recall']:.3f}")
    col4.metric(T['met_f1'],     f"{report[T['cm_left']]['f1-score']:.3f}")

    st.markdown('---')
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(T['chart_roc'])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1], [0,1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader(T['chart_cm'])
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=[T['cm_stayed'], T['cm_left']],
                    yticklabels=[T['cm_stayed'], T['cm_left']])
        ax.set_ylabel(T['cm_actual'])
        ax.set_xlabel(T['cm_predicted'])
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader(T['chart_fi'])
        fi = pd.DataFrame({
            'feature':    feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(fi['feature'], fi['importance'], color='steelblue')
        ax.set_xlabel(T['fi_xlabel'])
        st.pyplot(fig)
        plt.close()

    with col_right2:
        st.subheader(T['chart_dist'])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6,
                color='steelblue', label=T['dist_stayed'])
        ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6,
                color='salmon', label=T['dist_left'])
        ax.axvline(0.3, color='orange', linestyle='--', label=T['thr_30'])
        ax.axvline(0.6, color='red',    linestyle='--', label=T['thr_60'])
        ax.set_xlabel(T['dist_xlabel'])
        ax.set_ylabel(T['dist_ylabel'])
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.subheader(T['report_title'])
    df_report = pd.DataFrame(report).transpose().round(3)
    st.dataframe(df_report, width='stretch')


elif page == 'csv':
    st.title(T['csv_title'])
    st.markdown(T['csv_subtitle'])

    st.subheader(T['csv_template'])
    template = pd.DataFrame([{
        'age': 30, 'department': 'IT', 'position': 'Аналитик',
        'salary': 800, 'got_raise': 1, 'raise_count': 1,
        'months_since_last_raise': 6, 'has_training': 1,
        'trainings_count': 2, 'months_since_last_training': 5,
        'performance': 3.5, 'enps': 40,
        'avg_work_hours': 8.0, 'attendance_rate': 0.95, 'late_rate': 0.10
    }])
    st.dataframe(template)
    csv_template = template.to_csv(index=False).encode('utf-8-sig')
    st.download_button(T['csv_download_t'], csv_template, 'template.csv', 'text/csv')

    st.markdown('---')
    uploaded = st.file_uploader(T['csv_upload'], type='csv')

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.subheader(T['csv_loaded'])
        st.dataframe(df_input, width='stretch')

        try:
            df_encoded = df_input.copy()
            df_encoded['department'] = le_dept.transform(df_input['department'])
            df_encoded['position']   = le_pos.transform(df_input['position'])

            X_batch = df_encoded[feature_names]
            probs   = model.predict_proba(X_batch)[:, 1]

            df_result = df_input.copy()
            df_result['risk_score_%'] = (probs * 100).round(1)
            df_result['risk_level']   = pd.cut(
                probs, bins=[0, 0.3, 0.6, 1.0],
                labels=[T['risk_low'], T['risk_mid'], T['risk_high']]
            )
            df_result = df_result.sort_values('risk_score_%', ascending=False)

            st.markdown('---')
            st.subheader(T['csv_results'])
            st.dataframe(
                df_result[['department', 'position', 'risk_score_%', 'risk_level']]
                .reset_index(drop=True), width='stretch'
            )

            csv_result = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button(T['csv_download_r'], csv_result,
                               'attrition_results.csv', 'text/csv')

        except Exception as e:
            st.error(T['csv_error'].format(e))
