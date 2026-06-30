# HR Attrition Predictor

Streamlit-дашборд для прогнозирования риска увольнения сотрудников на основе HR-данных и данных СКУД (учёт посещаемости). Модель — XGBoost, с объяснением решений через SHAP.

## Структура проекта

```
.
├── app.py                  # Основное Streamlit-приложение
├── data/
│   ├── employees.csv       # База сотрудников
│   └── skud.csv            # Данные СКУД (посещаемость)
├── model/
│   ├── attrition_model.pkl       # Обученная XGBoost-модель
│   ├── le_department.pkl         # LabelEncoder для отделов
│   ├── le_position.pkl           # LabelEncoder для должностей
│   ├── feature_names.pkl         # Порядок признаков модели
│   ├── X_train.pkl, X_test.pkl   # Train/test выборки
│   └── y_test.pkl, y_pred.pkl, y_prob.pkl  # Результаты на test
├── requirements.txt
└── .gitignore
```

## Возможности

- **HR показатели** — общий обзор по компании: численность, средний стаж, eNPS, распределение по уровню риска, текучка по отделам.
- **Анализ сотрудника** — индивидуальный прогноз риска увольнения с SHAP-объяснением и рекомендациями.
- **Загрузка CSV** — пакетный анализ списка сотрудников из загруженного файла.
- **Метрики модели** — ROC-AUC, accuracy, recall, confusion matrix, важность признаков.

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Деплой на Streamlit Community Cloud

1. Зайти на [share.streamlit.io](https://share.streamlit.io)
2. New app → выбрать этот репозиторий, ветку `main`, файл `app.py`
3. Deploy
