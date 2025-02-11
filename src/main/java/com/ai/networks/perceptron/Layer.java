package com.ai.networks.perceptron;

/**
 * Класс Layer хранит нейроны определенного слоя, а также нейрон смещения
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public class Layer {
    /**
     * neurons - массив {@code Neuron}, хранящий основные нейроны
     */
    protected Neuron[] neurons;

    /**
     * b - {@code Neuron} смещения
     */
    protected Neuron biasNeuron;

    /**
     * Пустой конструктор, использующийся в наследнике для предотвращения повторного создания
     * @see #Layer(int, int)
     */
    public Layer(){}

    /**
     * Конструктор с параметрами, использующийся для инициализации слоя
     * @param length количество нейронов без нейрона смещения
     * @param nexLen количество нейронов следующего слоя
     * @see #Layer()
     */
    public Layer(int length, int nexLen){
        neurons = new Neuron[length];
        for(int i = 0; i < length; i++){
            neurons[i] = new Neuron(nexLen);
        }
        biasNeuron = new Neuron(nexLen);
    }

    /**
     * normalize - метод для активации значений в нейронах слоя.
     * Используется после подсчета значений у слоя.
     */
    public void normalize(){
        for (Neuron neuron : neurons) {
            neuron.normalizeRes();
        }
    }

    /**
     * setInput - метод для установки значений у входного слоя
     * <p>
     * Примечание: ожидается, что значния нормализованы
     * </p>
     * @param in массив {@code Double} - входные значения
     */
    public void setInput(Double[] in){
        if(in.length != neurons.length) return;
        for (int i = 0; i < in.length; i++) {
            neurons[i].setResult(in[i]);
            neurons[i].setNormResult(in[i]);
        }
    }

    /**
     * getNormResult - метод для получения нормального значения у нейрона
     * @param i индекс нейрона
     * @return {@code double} - нормальное значение
     * @throws IndexOutOfBoundsException если индекс выходит за пределы {@code neurons}
     * @see #getNormResult()
     */
    public double getNormResult(int i){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        return neurons[i].getNormResult();
    }

    /**
     * getNormResult - метод для получения нормального значения у нейронов слоя
     * @return массив {@code Double} нормальных значений нейронов
     * @see #getNormResult(int)
     */
    public Double[] getNormResult(){
        Double[] res = new Double[neurons.length];
        for(int i = 0; i < neurons.length; i++){
            res[i] = neurons[i].getNormResult();
        }
        return res;
    }

    /**
     * getWeight - метод для получения веса между нейронами i-ым этого и j-ым следующего слоя
     * @param i индекс нейрона текущего слоя
     * @param j индекс нейрона следующего слоя
     * @return {@code double} - вес
     * @throws IndexOutOfBoundsException если хотябы один индекс выходит за пределы
     * @see #getWeightB(int)
     */
    public double getWeight(int i, int j){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        if(j < 0 || j >= neurons[i].getLenWeight()) throw new IndexOutOfBoundsException(j);
        return neurons[i].getWeight(j);
    }

    /**
     * getWeightB - метод для получения веса между нейроном смщения i-ым нейроном следующего слоя
     * @param i индекс нейрона следующего слоя
     * @return {@code double} - вес
     * @throws IndexOutOfBoundsException если индекс выходит за пределы следующего слоя
     * @see #getWeight(int, int)
     */
    public double getWeightB(int i){
        if(i < 0 || i >= biasNeuron.getLenWeight()) throw new IndexOutOfBoundsException(i);
        return biasNeuron.getWeight(i);
    }

    /**
     * addResult - метод для добавления значения к результату у нейрона
     * @param i индекс нейрона
     * @param add добавляемое значение
     * @throws IndexOutOfBoundsException если индекс выходит за пределы {@code neurons}
     */
    public void addResult(int i, double add){
        if(i < 0 || i >= neurons.length) throw new IndexOutOfBoundsException(i);
        neurons[i].addRes(add);
    }

    /**
     * getLength - метод, использующийся для получения количества основных нейронов
     * @return количество нейронов слоя
     */
    public int getLength(){
        return neurons.length;
    }

    /**
     * setZero - метод для установки значений всех нейронов в 0.
     * Используется как очистка значений от прошлых подсчетов.
     */
    public void setZero(){
        for (Neuron neuron : neurons){
            neuron.setResult(0);
        }
    }

    /**
     * setWeight - метод для установки весов для заданного нейрона
     * @param weights массив {@code String}, содержащий значения весов
     * @param i индекс нейрона
     * @return {@code true} при успешной установке значений
     * @see #setWeightB(String[])
     */
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

    /**
     * setWeight - метод для установки весов для нейрона смещения
     * @param weights массив {@code String}, содержащий значения весов
     * @return {@code true} при успешной установке значений
     * @see #setWeight(String[], int)
     */
    public boolean setWeightB(String[] weights){
        try{
            for(int j = 0; j < weights.length; j++){
                biasNeuron.setWeight(Double.parseDouble(weights[j]), j);
            }
        } catch (NumberFormatException e) {
            return false;
        }
        return true;
    }
}
