package com.ai.networks;

import java.io.PrintStream;

/**
 * Интерфейс Training основан на {@code Network} и определяет дополнительное поведение любой обучающей нейронной сети
 * @param <IN> тип входных значений
 * @param <OUT> тип выходных значений
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public interface Training<IN, OUT> extends Network<IN, OUT>{

    /**
     * training - метод, запускающий процесс обучения нейронной сети на протяжении заданного количества эпох.
     * Предполагается, что обучение будет происходить в параллельном потоке.
     * @param dataset реализация {@code Teacher}, хранящая выборку
     * @param epochs количество эпох
     * @return {@code int} - результат запуска. При успшном - 0. При неудаче - 1.
     */
    int training(Teacher<IN, OUT> dataset, int epochs);

    /**
     * printingStatus - метод, запускающий процесс вывода на экран текущего состояния нейронной сети.
     * @param printStream поток вывода информации, необходим для корректного отображения символов.
     * @return {@code int} - результат запуска. При успшном - 0. При неудаче - 1.
     */
    int printingStatus(PrintStream printStream);

    /**
     * stop - метод, позволяющий остановить обучение
     */
    void stop();

    /**
     * getThread - метод, возвращающий поток, в котором происходит обучение
     * @return {@code Thread} обучения
     */
    Thread getThread();

    /**
     * isTraining - метод, возвращающий состояние процесса.
     * @return При запущенном процессе возвращает true.
     */
    boolean isTraining();

    /**
     * save - метод, позволяющий сохранить данные о нейронной сети в файл
     * @param file {@code String} - путь до файла
     * @throws RuntimeException при неудаче сохранить в файл
     * @see #save()
     */
    void save(String file);

    /**
     * save - метод, позволяющий обновить данные о нейронной сети в файл, если тот уже был использован
     * @throws RuntimeException при неудаче сохранить в файл
     * @see #save(String file)
     */
    void save();

    /**
     * getDeltaOfInputLayer - метод, использующийся для обучения последовательных нейронных сетей
     * @return массив {@code double} - дельты входного слоя
     */
    double[] getDeltaOfInputLayer();

    /**
     * getErrors - метод для получения ошибок(отклонений от ожидаемого значения) нейронной сети
     * @return массив {@code double}: [средняя ошибка, максимальная ошибка]
     */
    double[] getErrors();
}
