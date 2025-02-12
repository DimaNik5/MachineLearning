package com.ai.networks.perceptron;

/**
 * Класс Neuron хранит значения, а также веса к нейронам следующего слоя
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public class Neuron {
    protected double result;
    protected double normResult;
    protected double[] weight;

    /**
     * Метод активации нейрона, представленный в виде функции сигмоиды
     * @param x значения для активации
     * @return нормализованное значение
     */
    protected double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Инициализирующий конструктор с параметрами
     * @param countWeight количество нейронов в следующем слое
     * @see #Neuron()
     */
    public Neuron(int countWeight){
        weight = new double[countWeight];
    }

    /**
     * Пустой конструктор, использующийся в наследнике для предотвращения повторного создания
     * @see #Neuron(int)
     */
    public Neuron(){}

    /**
     * Метод для активации нейрона
     */
    public void normalizeRes(){
        normResult = sigmoid(result);
    }

    public void addRes(double add){
        result += add;
    }

    /**
     * Метод, использующийся для установки нормального значения
     * @param result новое нормаьное значение
     * @see #setResult(double)
     */
    public void setNormResult(double result){
        normResult = result;
    }

    /**
     * Метод, использующийся для установки значения
     * @param result новое нормаьное значение
     * @see #setNormResult(double)
     */
    public void setResult(double result){
        this.result = result;
    }

    /**
     * Метод для получения нормального значения
     * @return нормальное значение нейрона
     */
    public double getNormResult() {
        return normResult;
    }

    /**
     * Метод для получения веса к i-ому нейрону следующего слоя
     * @param i индекс нейрона следующего слоя
     * @return значение веса
     * @throws IndexOutOfBoundsException если индекс выходит за пределы
     */
    public double getWeight(int i){
        if(i < 0 || i >= weight.length) throw new IndexOutOfBoundsException(i);
        return weight[i];
    }

    /**
     * Метод для получения количества нейронов в следующем слое
     * @return количество весов
     */
    public double getLenWeight(){
        return weight.length;
    }

    /**
     * Метод для установки значения конкретного веса
     * @param w значение веса
     * @param i индекс веса
     * @throws IndexOutOfBoundsException если индекс выходит за пределы
     */
    public void setWeight(double w, int i){
        if(i >= weight.length) throw new IndexOutOfBoundsException(i);
        weight[i] = w;
    }
}
