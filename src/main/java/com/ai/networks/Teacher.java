package com.ai.networks;

import java.util.List;

/**
 * Интерфейс Teacher определяет поведение любой выборки (датасет, учитель)
 * @param <IN> тип входных значений
 * @param <OUT> тип выходных значений
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public interface Teacher<IN, OUT> {

    /**
     * Метод, позволяющий загрузить новые значения выборки
     * <p>
     * Примечание: размер входной выборки и выборки с ожидаемыми значениями должны совпадать и не равны 0.
     * </p>
     * @param inputs двумерный массив {@code IN}
     * @param outputs двумерный массив {@code OUT}
     * @return {@code int} - результат загрузки.
     * @see #setData(List, List) 
     */
    int setData(IN[][] inputs, OUT[][] outputs);

    /**
     * Метод, позволяющий загрузить новые значения выборки
     * <p>
     * Примечание: размер входной выборки и выборки с ожидаемыми значениями должны совпадать и не равны 0.
     * </p>
     * @param inputs список массивов {@code IN}
     * @param outputs список массивов {@code OUT}
     * @return {@code int} - результат загрузки.
     * @see #setData(Object[][], Object[][])
     */
    int setData(List<IN[]> inputs, List<OUT[]> outputs);

    /**
     * Метод, позволяющий добавить значения выборки
     * <p>
     * Примечание: размер входной выборки и выборки с ожидаемыми значениями должны совпадать и не равны 0.
     * </p>
     * @param inputs двумерный массив {@code IN}
     * @param outputs двумерный массив {@code OUT}
     * @return {@code int} - результат загрузки. При удачной - 0. При неудачной - 1.
     * @see #addData(List, List)
     * @see #addData(Object[], Object[])
     */
    int addData(IN[][] inputs, OUT[][] outputs);

    /**
     * Метод, позволяющий добавить значения выборки
     * <p>
     * Примечание: размер входной выборки и выборки с ожидаемыми значениями должны совпадать и не равны 0.
     * </p>
     * @param inputs список массивов {@code IN}
     * @param outputs список массивов {@code OUT}
     * @return {@code int} - результат загрузки. При удачной - 0. При неудачной - 1.
     * @see #addData(Object[][], Object[][])
     * @see #addData(Object[], Object[])
     */
    int addData(List<IN[]> inputs, List<OUT[]> outputs);

    /**
     * Метод, позволяющий добавить одно значение выборки
     * @param inputs массив {@code IN}
     * @param outputs массив {@code OUT}
     * @return {@code int} - результат загрузки. При удачной - 0. При неудачной - 1.
     * @see #addData(Object[][], Object[][])
     * @see #addData(List, List)
     */
    int addData(IN[] inputs, OUT[] outputs);

    /**
     * Используется перед обучением для получения входных значений выборки
     * @return список массивов входных значений
     */
    List<IN[]> getIn();

    /**
     * Используется перед обучением для получения ожидаемых значений в выборке
     * @return список массивов выходных значений
     */
    List<OUT[]> getOut();

    /**
     * Используется перед {@code getIn} и {@code getOut}
     * @return {@code true}, если массивы входных и ожидаемых значений заданы и корректны
     */
    boolean checkData();
}
