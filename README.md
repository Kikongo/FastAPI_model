# FastAPI_model
В данной работе была обучена модель с помощью классической линейной регрессии.<br/>
По метрике R2 наилучший результат был 0.62. Его удалось достигнуть после добавления категориальных признаков к модели и применения L2 регуляризации.<br/>
Пришлось удалить столбец seats, так как размерности train датасета не совпадали с test после кодирования категориальных фичей. Было непонятно как с этим правильно бороться.<br/>
API реализована в файле app.py
