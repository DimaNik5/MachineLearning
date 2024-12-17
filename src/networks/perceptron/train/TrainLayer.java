package networks.perceptron.train;

import networks.perceptron.Layer;

public class TrainLayer extends Layer {
    private TrainNeuron[] trainNeurons;
    private  TrainNeuron tb;

    public TrainLayer(int length) {
        super(length);
    }

    public TrainLayer(int length, int nexLen){
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

    // Установка новых весов
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

    // Получение списка дельт
    public double[] getDeltas(){
        double[] res = new double[trainNeurons.length];
        for(int i = 0; i < trainNeurons.length; i++){
            res[i] = trainNeurons[i].getDelta();
        }
        return res;
    }

    // Пропорциональной деление весов нейронов
    public void divWeight(double div){
        for(TrainNeuron neuronTrain : trainNeurons){
            neuronTrain.divWeight(div);
        }
    }


}
