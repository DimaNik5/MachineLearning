package networks.perceptron;

public class Neuron {
    protected double result;
    // Нормальное значение (0-1)
    protected double normResult;
    protected double[] weight;

    // Функция нормализации(активации)
    protected double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public Neuron(){}

    public void normalizeRes(){
        normResult = sigmoid(result);
    }

    public void addRes(double add){
        result += add;
    }

    // используется для установки входных данных
    public void setNormResult(double result){
        normResult = result;
    }

    public void setResult(double result){
        this.result = result;
    }

    public double getNormResult() {
        return normResult;
    }

    public double getWeight(int i){
        if(i >= weight.length) throw new IndexOutOfBoundsException(i);
        return weight[i];
    }

    public double getLenWeight(){
        return weight.length;
    }

    public void setWeight(double w, int i){
        if(i >= weight.length) throw new IndexOutOfBoundsException(i);
        weight[i] = w;
    }
}
