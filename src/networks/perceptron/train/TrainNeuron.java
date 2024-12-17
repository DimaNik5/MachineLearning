package networks.perceptron.train;

import networks.perceptron.Neuron;

import java.util.Random;

public class TrainNeuron extends Neuron {
    // Прошлые изменения весов (дельта весов)
    private double[] lastModWeight;
    private double delta;

    // Производная от функции активации
    private double derSigmoid(double res){
        return (1 - res) * res;
    }

    public TrainNeuron(int countWeight){
        weight = new double[countWeight];
        lastModWeight = new double[countWeight];
        for(int i = 0; i < countWeight; i++){
            Random random = new Random();
            weight[i] = random.nextDouble(2) - 1;
        }
    }

    public TrainNeuron(){}

    // Вычисленное значение (разница от идеала/суииа произведений) домнажается на производную от функции
    public double setDelta(double delta){
        return (this.delta = delta * derSigmoid(normResult));
    }

    // Деление весов
    public void divWeight(double div){
        for(int i = 0; i < weight.length; i++)
            weight[i] /= div;
    }

    public double getDelta(){
        return delta;
    }

    // Установка весов
    public double setDeltaWeight(double[] delta, double speed, double alpha){
        double max = 0;
        // Проход п дельтам следующего слоя (по весам)
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
