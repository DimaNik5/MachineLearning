package networks.perceptron;

public class Layer {
    protected Neuron[] neurons;
    // Нейрон смещения
    protected Neuron b;

    public Layer(){};
    public Layer(int length){
        neurons = new Neuron[length];
        for(int i = 0; i < length; i++){
            neurons[i] = new Neuron();
        }
        b = new Neuron();
    }

    // Нормализация(активация)
    public void normalize(){
        for (Neuron neuron : neurons) {
            neuron.normalizeRes();
        }
    }

    // Установка входных значений
    public void setInput(Double[] in){
        for (int i = 0; i < in.length; i++) {
            neurons[i].setResult(in[i]);
            neurons[i].setNormResult(in[i]);
        }
    }

    // Получение нормального значения i-того нейрона
    public double getNormResult(int i){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        return neurons[i].getNormResult();
    }

    // Получение списка нормальных значений нейронов
    public Double[] getNormResult(){
        Double[] res = new Double[neurons.length];
        for(int i = 0; i < neurons.length; i++){
            res[i] = neurons[i].getNormResult();
        }
        return res;
    }

    // Получение веса от i к j нейрону
    public double getWeight(int i, int j){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        if(j < 0 || j >= neurons[i].getLenWeight()) throw new IndexOutOfBoundsException(j);
        return neurons[i].getWeight(j);
    }

    // Получение веса нейрона смещения
    public double getWeightB(int i){
        return b.getWeight(i);
    }

    // добавление к результату i-того нейрона
    public void addResult(int i, double add){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        neurons[i].addRes(add);
    }

    public int getLength(){
        return neurons.length;
    }

    // Установка результатов нейронов в 0
    public void setZero(){
        for (Neuron neuron : neurons){
            neuron.setResult(0);
        }
    }

    // Установка весов для заданного нейрона
    public boolean setWeight(String[] weights, int i){
        if(i < 0 || i >= neurons.length) return false;
        try{
            for(int j = 0; j < weights.length; j++){
                neurons[i].setWeight(Double.parseDouble(weights[j]), j);
            }
        } catch (NumberFormatException e) {
            return false;
        }
        return true;
    }

    // Установка весов для нейрона смещения
    public boolean setWeightB(String[] weights){
        try{
            for(int j = 0; j < weights.length; j++){
                b.setWeight(Double.parseDouble(weights[j]), j);
            }
        } catch (NumberFormatException e) {
            return false;
        }
        return true;
    }
}
