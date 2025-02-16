package com.ai.networks.perceptron.train.backpropagation;

import com.ai.networks.perceptron.Neuron;

import java.util.Random;

/**
 * Класс TrainNeuron дополняет сущность {@code Neuron} возможностью обучения.
 * @author Никифоров Дмитрий
 * @since 1.0
 */
final class TrainNeuron extends Neuron {
    private double[] lastModWeight;
    private double delta;

    /**
     * Метод получения значения производной функции активации (сигмоиды) в конкретной точке
     * @param res - нормализованое значение
     * @return значение производной
     */
    private double derSigmoid(double res){
        return (1 - res) * res;
    }

    /**
     * Конструктор для создания нейрона со связями
     * @param countWeight количество нейорнов в следующем слое
     * @see #TrainNeuron()
     */
    TrainNeuron(int countWeight){
        weight = new double[countWeight];
        lastModWeight = new double[countWeight];
        for(int i = 0; i < countWeight; i++){
            Random random = new Random(System.currentTimeMillis());
            weight[i] = random.nextDouble() * 2 - 1;
        }
    }

    /**
     * Конструктор для создания нейронов выходного слоя
     * @see #TrainNeuron(int)
     */
    TrainNeuron(){}

    /**
     * Метод для установки дельты, которая домножатся на производную
     * @param delta значение дельты
     */
    public double setDelta(double delta){
        return (this.delta = delta * derSigmoid(normResult));
    }

    /**
     * Метод для деления всех весов нейрона
     * @param div знаменатель
     */
    public void divWeight(double div){
        for(int i = 0; i < weight.length; i++)
            weight[i] /= div;
    }

    /**
     * Метод для получения дельты нейрона
     * @return дельта нейрона
     */
    public double getDelta(){
        return delta;
    }

    /**
     * Метод для подсчета и изменения весов нейрона по дельтам следующего слоя
     * @param delta массив дельт следующего слоя
     * @param speed скорость обучения
     * @param alpha момент градиентного спуска
     * @return максимальный новый вес
     */
    public double setDeltaWeight(double[] delta, double speed, double alpha){
        double max = 0;
        // Проход по дельтам следующего слоя (по весам)
        for(int i = 0; i < delta.length; i ++){
            // result * delta[i] - GRADIENT
            // Изменение весов
            double wd = delta[i] * speed * result + alpha * lastModWeight[i];
            weight[i] += wd;
            if(weight[i] > max) max = weight[i];
            if(Math.abs(weight[i]) < 0.00001) weight[i] = 0;
            lastModWeight[i] = wd;
        }
        return max;
    }
}
