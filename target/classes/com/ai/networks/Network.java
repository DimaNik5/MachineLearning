package com.ai.networks;

/**
 * Интерфейс Network определяет поведение любой нейронной сети
 * @param <IN> тип входных значений
 * @param <OUT> тип выходных значений
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public interface Network<IN, OUT> {

    /**
     * Метод, позволяющий загружать данные о неронной сети из файла.
     * @param fileName {@code String}, путь до файла.
     * @return {@code int} - результат загрузки. При успешной загрузке - 0. При неудаче - 1.
     * @throws NumberFormatException если в файле нечисловой формат данных
     * @throws RuntimeException при неудаче считать из файла
     */
    int loadFromFile(String fileName);

    /**
     * Метод, позволяющий загружать данные о неронной сети из строки.
     * @param content {@code String}, данные о неронной сети.
     * @return {@code int} - результат загрузки. При успешной загрузке - 0. При неудаче - 1.
     * @throws NumberFormatException если в строке нечисловой формат данных
     */
    int loadFromString(String content);

    /**
     * Метод, загружающий входные данные в нейронную сеть
     * @param input массив {@code IN}, входные данные
     */
    void setInput(IN[] input);

    /**
     * Метод считающий результат
     */
    void counting();

    /**
     * Метод, возвращающий посчитанные данные нейронной сетью
     * @return массив {@code OUT} - выходные данные
     */
    OUT[] getOutput();

    /**
     * Метод, возвращающий тип используемой нейронной сети
     * @return {@code NetworkModels} - тип
     */
    NetworkModels getModel();
}
