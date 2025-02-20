package com.ai.networks.perceptron.train.backpropagation;

import com.ai.networks.perceptron.Layer;

/**
 * Класс TrainLayer дополняет сущность {@code Layer} возможностью обучения.
 * @author Никифоров Дмитрий
 * @since 1.0
 */
final class TrainLayer extends Layer {
    /**
     * trainNeurons - массив {@code TrainNeuron} для обучения
     */
    private final TrainNeuron[] trainNeurons;
    /**
     * tb - нейрон смещения с возможность обучения
     */
    private final TrainNeuron tb;

    /**
     * Конструктор для создания нового слоя
     * @param length количество нейронов без учета нейрона смещния
     * @param nexLen количество нейронов в следующем слое
     */
    TrainLayer(int length, int nexLen){
        trainNeurons = new TrainNeuron[length];
        if(nexLen > 0){
            for(int i = 0; i < length; i++){
                trainNeurons[i] = new TrainNeuron(nexLen);
            }
            tb = new TrainNeuron(nexLen);
        }
        else {
            for(int i = 0; i < length; i++){
                trainNeurons[i] = new TrainNeuron();
            }
            tb = new TrainNeuron();
        }
        b = tb;
        neurons = trainNeurons;
    }

    /**
     * setDelta считает и устанавливает дельты (разница) нейронам этого слоя по дельтам следующего - обратное распростронение ошибки (дельты)
     * @param ideal массив дельт следующего слоя
     * @param last флаг о том, что это выходной слой
     */
    public Double[] setDelta(Double[] ideal, boolean last){
        Double[] res = new Double[trainNeurons.length];
        for(int i = 0; i < trainNeurons.length; i++){
            // Выходной слой
            if(last){
                // Разница между идеалом
                res[i] = trainNeurons[i].setDelta(ideal[i] - trainNeurons[i].getNormResult());
            }
            else{
                // Сумма произведений весов на соот. дельты следующего слоя
                double sum = 0;
                for(int j = 0; j < ideal.length; j++){
                    sum += trainNeurons[i].getWeight(j) * ideal[j];
                }
                res[i] = trainNeurons[i].setDelta(sum);
            }
        }
        return res;
    }

    /**
     * setDeltaWeight - метод для установки новых весов по дельтам следующего слоя
     * @param delta массив дельт следующего слоя
     * @param speed скорость обучения
     * @param alpha момент градиентного спуска
     * @return максимальный новый вес в этом слое
     */
    public double setDeltaWeight(double[] delta, double speed, double alpha){
        double max = 0, t;
        for(TrainNeuron neuronTrain : trainNeurons){
            t = neuronTrain.setDeltaWeight(delta, speed, alpha);
            if(t > max) max = t;
        }
        t = tb.setDeltaWeight(delta, speed, alpha);
        if(t > max) max = t;
        return max; // максимальный новый вес
    }

    /**
     * getDelta - метод для получения дельт нейронов
     * @return массив дельт у нейронов в этом слое
     */
    public double[] getDeltas(){
        double[] res = new double[trainNeurons.length];
        for(int i = 0; i < trainNeurons.length; i++){
            res[i] = trainNeurons[i].getDelta();
        }
        return res;
    }

    /**
     * divWeight - метод для ропорциональной деления весов нейронов, если вес какой-либо связи превысил максимальный
     */
    public void divWeight(double div){
        for(TrainNeuron neuronTrain : trainNeurons){
            neuronTrain.divWeight(div);
        }
    }
}
