package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Teacher;
import com.ai.networks.Tokens;
import com.ai.networks.Training;
import com.ai.networks.convolutional.Convolution;
import com.ai.networks.convolutional.Layer;
import com.ai.networks.convolutional.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;

/**
 * Класс TrainConvolutionBackpropagation дополняет сущность {@code Convolution} возможностью обучения.
 * <p>
 * Примечание: {@code Convolution} является часть сверточной нейронной сети, поэтому нет возможности получить
 * конечные значение, тем самым нет возможности обучения по выборке.
 * </p>
 * <h2>Обучение</h2>
 * <p>
 * Для обучения в данном классе используется метод обратного распространения (Backpropagation).
 * Для этого используется метод нахождения дельт и стохастический метод обновления весов.
 * Для изменения весов в лучшую сторону используется градиентный спуск,
 * а чтобы не останавится в первом локальном минимуме, используется момент.
 * </p><br>
 * <h3>Алгоритм:</h3>
 * <ul>
 *
 * <li>
 * Подсчет результата по входным значеням.
 * </li>
 *
 * <li>
 * Плучение дельты выходного слоя.
 * </li>
 *
 * <li>
 * Распространение дельты на все остальные слои.
 * Дельтой для фильтра является свертка дельт соответсвенного выходного слоя с соответсвенным каналом входного слоя.
 * Дельтой канала входного слоя является сумма сверток матриц дельт выходного слоя, дополненного нулями,
 * с соответсвенным перевернутым на 180 фильтром такого же канала.
 * </li>
 *
 * <li>
 * Подсчитывание изменение весов у фильтров.
 * Изменение веса равняется сумме поризведений градиента на скорость обучения и момента градиентного спуска на последнее изменение.
 * После чего, посчитанное изменение добавляется и проверяется на превышение максимального.
 * </li>
 *
 * <li>
 * Если вес превышает допустимое значение, то все веса фильтра уменьшаются пропорционально.
 * </li>
 *
 * </ul>
 * @author Никифоров Дмитрий
 * @since 1.1
 */
public class TrainConvolutionBackpropagation extends Convolution implements Training<Matrix, Matrix> {

    private TrainLayer[] trainLayers;
    private double speed;
    /**
     * alpha - момент градиентного спуска.
     * Он позволяет выбираться из локальных минимумов.
     */
    private double alpha;
    private double maxWeight;
    private String fileName;

    /**
     * Конструктор для создания новой нейронной сети.<br>
     * Структура: {
     *     <ul>
     *         <li>высота входного слоя</li>
     *         <li>ширина входного слоя</li>
     *         <li>количество входных каналов</li>
     *         <li>количество фильтров</li>
     *         <li>размер фильтра</li>
     *     </ul>
     * }<br>
     * Последние два пункта повторяются в зависимости от количества слоев.
     * @param layers структура
     * @param speed скорость обучения
     * @param alpha момент
     * @param maxWeight максимальный вес
     * @see #TrainConvolutionBackpropagation(String, double, double, double)
     */
    public TrainConvolutionBackpropagation(int[] layers, double speed, double alpha, double maxWeight){
        trainLayers = new TrainLayer[(layers.length - 3) / 2];
        this.layers = trainLayers;
        for (int i = 0; i < (layers.length - 3) / 2; i++) {
            trainLayers[i] = new TrainLayer();
        }
        this.speed = speed;
        this.alpha = alpha;
        this.maxWeight = maxWeight;
        trainLayers[0].createNewLayer(layers[0], layers[1], layers[2], layers[3], layers[4]);
        for (int i = 1; i < trainLayers.length; i++) {
            trainLayers[i].createNewLayer(trainLayers[i - 1].getOut(), layers[3 + i * 2], layers[4 + i * 2]);
        }
    }

    /**
     * Конструктор для загрузки сети из строки
     * @param content строка, содержащая информацию про сеть
     * @param speed скорость обучения
     * @param alpha момент
     * @param maxWeight максимальный вес
     * @see #TrainConvolutionBackpropagation(int[], double, double, double)
     */
    public TrainConvolutionBackpropagation(String content, double speed, double alpha, double maxWeight){
        this.speed = speed;
        this.alpha = alpha;
        this.maxWeight = maxWeight;
        loadFromString(content);
    }

    @Override
    public String getContent() {
        StringBuilder cont = trainLayers[0].getContent();
        for (int i = 1; i < trainLayers.length; i++) {
            cont.append(Tokens.SEP_OF_LAYERS);
            trainLayers[i].getContent(cont);
        }
        return cont.toString();
    }

    @Override
    public void save(String file) {
        fileName = file;
        save();
    }

    @Override
    public void save() {
        if(fileName.isEmpty()) return;
        try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
            StringBuilder cont = trainLayers[0].getContent();
            for (int i = 1; i < trainLayers.length; i++) {
                cont.append(Tokens.SEP_OF_LAYERS);
                trainLayers[i].getContent(cont);
            }
            bw.write(cont.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public double[] getDeltaOfInputLayer() {
        int l = 0;
        for(Matrix m : trainLayers[0].getDeltasOfInput()){
            l += m.getH() * m.getW();
        }
        double[] res = new double[l];
        int k = 0;
        for(Matrix m : trainLayers[0].getDeltasOfInput()){
            double[][] mat = m.getMatrix();
            for (double[] doubles : mat) {
                for (double aDouble : doubles) {
                    res[k++] = aDouble;
                }
            }
        }
        return res;
    }

    @Override
    public void setDeltaOfOutputLayer(double[] deltas) {
        int l = trainLayers.length;
        int h = trainLayers[l - 1].getOut()[0].getH();
        int w = trainLayers[l - 1].getOut()[0].getW();
        trainLayers[l - 1].setDeltas(Matrix.getMatricesFromArray(deltas, trainLayers[l - 1].getOut().length, h, w));
        trainLayers[l - 1].correcting(speed, alpha, maxWeight);
        for (int i = l - 2; i >= 0; i--) {
            trainLayers[i].setDeltas(trainLayers[i + 1].getDeltasOfInput());
            trainLayers[i].correcting(speed, alpha, maxWeight);
        }
    }

    @Override
    public int loadFromString(String content) {
        try{
            String[] l = content.split(Tokens.SEP_OF_LAYERS);
            trainLayers = new TrainLayer[l.length];
            layers = trainLayers;
            trainLayers[0] = new TrainLayer();
            trainLayers[0].loadFirstFromString(l[0]);
            for(int i = 1; i < l.length; i++){
                trainLayers[i] = new TrainLayer();
                trainLayers[i].loadFromString(l[i], trainLayers[i - 1].getOut());
            }
        }catch (Exception e) {
            return 1;
        }
        return 0;
    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public double[] getErrors() {
        return null;
    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public int training(Teacher<Matrix, Matrix> dataset, int epochs) {
        return 1;
    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public int printingStatus(PrintStream printStream) {
        return 1;
    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public void stop() {

    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public Thread getThread() {
        return null;
    }

    /**
     * В данной нейронной сети не используется
     */
    @Override
    public boolean isTraining() {
        return false;
    }

}
