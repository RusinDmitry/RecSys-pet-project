# Рекомендательная система постов для социальной сети
Реализован сервис, который для каждого пользователя в любой момент времени вовзращает набор постов, которые пользователю покажут в его ленте
___

Сервис учитывает предыдущие пожелания (лайки) каждого пользователя

## Структура
- В папке Models хранятся последние 3 модели с наивысшим качеством, используются для реализации А\В тестов
- app.py содержит в себе endpoint'ы и функции предобработки входящих запросов
- table_*.py описание таблиц в виде экзепляров класса
- schema.py схемы на pydantix
- Training_model.ipybn обучение моделей

## Запуск
Для запуска сервиса нужно ввести в терминале

    uvicorn app:app --reload --port 8899 

## Этапы работ
- Предварительная обработка данных;
- Удаление аномальных значений;
- Удаление параметров, имеющих количество пропусков более 50%;
- Удаление неинформативных признаков с одинаковыми значениями;
- Заполнение пустых значений;
- Балансировка данных;
- Масштабирование данных;
- Построение и обучение пайплайна на основе catboost
- Реализация веб-интерфейса
- Подготовка функционала для обработки тестовых данных в модель на продакт версии
- Добавление А/Б тестов
  
## Требования
Сервису необходимы следующие пакеты:
- python (3.11);
- fastapi;
- uvicorn;
- picke;
- haslib; 
- pandas;
