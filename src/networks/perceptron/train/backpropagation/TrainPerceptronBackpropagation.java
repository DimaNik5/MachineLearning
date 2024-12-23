package networks.perceptron.train.backpropagation;

import networks.Teacher;
import networks.Tokens;
import networks.Training;
import networks.perceptron.Perceptron;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Класс TrainPerceptronBackpropagation дополняет сущность {@code Perceptron} возможностью обучения.
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
 * Нахождение дельты выходного слоя - разница между результатом нейронной сети и ожидаемого значения.
 * </li>
 *
 * <li>
 * Распространение дельты на все остальные слои.
 * Дельтой не выходного нейрона является сумма произведений дельт нейронов из следующего слоя на вес между соответсвенным нейроном и текущим.
 * Дополнительно дельта домножается на производную функции активации.
 * </li>
 *
 * <li>
 * Подсчитывание изменение весов у нейорнов.
 * Изменение веса равняется сумме поризведений градиента на скорость обучения и момента градиентного спуска на последнее изменение.
 * После чего, посчитанное изменение добавляется и находится самый большой вес.
 * </li>
 *
 * <li>
 * Если самый большой вес превышает допустимое значение, то все веса уменьшаются пропорционально.
 * </li>
 *
 * <li>
 * Весь алгоритм проходит по всей выборке и повторяется в течении количества эпох или пока не будет прерван.
 * </li>
 * </ul>
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public class TrainPerceptronBackpropagation extends Perceptron implements Training<Double, Double> {
    /**
     * trainLayers - массив {@code TrainLayer} для обучения
     */
    private TrainLayer[] trainLayers;
    /**
     * speed - коэффициент скорости обучения.
     * Чем меньше величина, тем дольше обучение, но меньше шансов пропустить локальный минимум.
     */
    private double speed;
    /**
     * alpha - момент градиентного спуска.
     * Он позволяет выбираться из локальных минимумов.
     */
    private double alpha;
    /**
     * maxWeight - допустимое максимальное значение веса
     */
    private double maxWeight;
    /**
     * fileName - путь до файла, если он есть, где сохранен перцептрон
     */
    private String fileName;

    /**
     * middleError - текущая средняя ошибка перцептрона
     */
    private double middleError = 0;
    /**
     * maxError - текущая максимальная ошибка перцептрона
     */
    private double maxError = 0;

    /**
     * isTraining - флаг, означающий процесс обучения
     */
    private boolean isTraining = false;
    /**
     * isPrinting - флаг, означающий процесс вывода состояния
     */
    private boolean isPrinting = false;
    /**
     * countDone - количество пройденных эпох
     */
    private int countDone;
    /**
     * epochs - количество эпох, в течение которых происходит обучение
     */
    private int epochs = 1;
    /**
     * in - выборка входных значений
     */
    private List<Double[]> in;
    /**
     * out - выборка ожидаемых значений
     */
    private List<Double[]> out;

    /**
     * training - отдельный поток для обучения
     */
    private final Thread training;
    /**
     * printing - отдельный поток для вывода состояния
     */
    private final Thread printing;

    /**
     * printStream - {@code PrintStream} для вывода русских символов в консоль
     */
    private PrintStream printStream;


    // Инициализация потоков
    {

        training = new Thread(() -> {
            countDone = 0;
            double mide, maxe;
            // Тренировка по эпохам
            for (int k = 0; k < epochs; k++){
                mide = 0;
                maxe = 0;
                // Проход по тестам
                for (int i = 0; i < in.size(); i++) {
                    setInput(in.get(i));
                    // Подсчет результата
                    counting();
                    double t = 0;
                    for(int j = 0; j < out.get(i).length; j++) {
                        double d = Math.abs(out.get(i)[j] - getOutput()[j]);
                        t += d;
                        maxe = Math.max(d, maxe);
                    }
                    mide += t / out.get(i).length;
                    // Корректировка весов
                    correction(out.get(i));
                }
                mide /= in.size();
                middleError = mide;
                maxError = maxe;
                countDone++;
                if(!isTraining) break;
            }
            isPrinting = false;
        });

        printing = new Thread(() -> {

            String proc = "Пройдено эпох ";
            String me = "\nСредняя ошибка :";
            String maxe = "\nМаксимальная ошибка: ";
            StringBuilder res;
            boolean last = true;
            while(isPrinting || last){
                System.out.println(new String(new char[25]).replace("\0", "\r\n"));
                res = new StringBuilder(proc + countDone + '/' + epochs + " [");
                int part = (countDone * 10 / epochs) % 100;
                res.append("+".repeat(Math.max(0, part)));
                res.append("-".repeat(Math.max(0, 10 - part)));
                res.append(']');
                res.append(me).append(String.format("%.3f", middleError));
                res.append(maxe).append(String.format("%.3f", maxError));
                printStream.println(res);
                last = isPrinting;
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    /**
     * Конструктор для создания нового перцептрона по переданной структуре
     * @param lenLayer массив {@code int}, содержащий количество нейорнов в каждом слое. Длина массива (количество слоев) не должна быть меньше двух.
     * @param speed определяет скорость обучеения
     * @param alpha определяет момент градиентного спуска
     * @param maxWeight максимальный модуль значения веса
     * @see #TrainPerceptronBackpropagation(String, double, double, double)
     */
    public TrainPerceptronBackpropagation(int[] lenLayer, double speed, double alpha, double maxWeight){
        int len = lenLayer.length;
        if(len < 2) return;
        this.speed = speed;
        this.maxWeight = maxWeight;
        this.alpha = alpha;
        trainLayers = new TrainLayer[len];
        for(int i = 0; i < len; i++){
            if(lenLayer[i] < 0) return;
            trainLayers[i] = new TrainLayer(lenLayer[i], (i < len - 1 ? lenLayer[i + 1] : 0));
        }
        layers = trainLayers;
    }

    /**
     * Конструктор для до обучения сохраненного перцептрона
     * @param fileName {@code String} - путь до файла, где сохранен перцептрон
     * @param speed определяет скорость обучеения
     * @param alpha определяет момент градиентного спуска
     * @param maxWeight максимальный модуль значения веса
     * @see #TrainPerceptronBackpropagation(int[], double, double, double)
     */
    public TrainPerceptronBackpropagation(String fileName, double speed, double alpha, double maxWeight){
        this.speed = speed;
        this.maxWeight = maxWeight;
        this.alpha = alpha;
        this.fileName = fileName;
        loadFromFile(fileName);
    }

    /**
     * correction - метод для корректировки всех весовобратным распространением ошибки
     * @param ideal массив ожидаемых значений
     */
    private void correction(Double[] ideal){
        Double[] curIdeal = ideal;
        // Получение дельт(разницы от идеала) для каждого слоя с передачей результатов к предыдущему
        for(int i = trainLayers.length - 1; i >= 0; i--){
            curIdeal = trainLayers[i].setDelta(curIdeal, i == trainLayers.length - 1);
        }
        boolean f = false; // Превышение максимального веса
        // Установка новых весов
        for(int i = 0; i < trainLayers.length - 1; i++){
            f = f || (Math.abs(trainLayers[i].setDeltaWeight(trainLayers[i + 1].getDeltas(), speed, alpha)) > maxWeight);
        }
        if(f){
            // Пропорциональное деление весов
            for(int i = 0; i < trainLayers.length - 1; i++){
                trainLayers[i].divWeight(maxWeight / 2);
            }
        }
    }

    @Override
    public int training(Teacher<Double, Double> dataset, int epochs) {
        if(!dataset.checkData()) return 1;

        this.epochs = epochs;

        in = new ArrayList<>(dataset.getIn().size());
        in.addAll(dataset.getIn());

        out = new ArrayList<>(dataset.getOut().size());
        out.addAll(dataset.getOut());

        if(!training.isAlive()){
            isTraining = true;
            training.start();
            return 0;
        }
        return 1;
    }

    @Override
    public int printingStatus(PrintStream ps) {
        printStream = Objects.requireNonNullElseGet(ps, () -> new PrintStream(System.out, true, StandardCharsets.UTF_8));
        if(!printing.isAlive()){
            isPrinting = true;
            printing.start();
            return 0;
        }
        return 1;
    }

    @Override
    public void stop() {
        if(isTraining) isTraining = false;
    }

    @Override
    public Thread getThread() {
        return training;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void save(String file) {

        if(fileName == null || !fileName.equals(file)) fileName = file;
        try(FileWriter fw = new FileWriter(fileName)) {
            // Запись количества слоев
            fw.write(trainLayers.length + Tokens.SEP_OF_ELEMENTS);
            // Запись количества нейронов в каждом слое
            for(TrainLayer layerTrain : trainLayers){
                fw.write(layerTrain.getLength() + Tokens.SEP_OF_ELEMENTS);
            }
            fw.write(Tokens.SEP_OF_LAYERS);

            // Проход по всем слоям, кроме выходного(нет весов)
            for(int k = 0; k < trainLayers.length - 1; k++){
                // Проход повсем нейронам слоя
                for(int i = 0; i < trainLayers[k].getLength(); i++){
                    // Проход повсем нейронам следующего слоя
                    for(int j = 0; j < trainLayers[k + 1].getLength(); j++){
                        // Запись весов от i к j нейрону
                        fw.write(trainLayers[k].getWeight(i, j) + Tokens.SEP_OF_ELEMENTS);
                    }
                    // Знак препинания
                    fw.write(Tokens.SEP_OF_OBJECTS);
                }
                // Запись весов нейрона смещения
                for(int j = 0; j < trainLayers[k + 1].getLength(); j++){
                    fw.write(trainLayers[k].getWeightB(j) + Tokens.SEP_OF_ELEMENTS);
                }
                fw.write(Tokens.SEP_OF_LAYERS);
            }
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void save() {
        if(!fileName.isEmpty()){
            save(fileName);
        }
    }

    @Override
    public double[] getErrors() {
        return new double[]{middleError, maxError};
    }

    @Override
    public int loadFromFile(String fileName) {
        try(BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            int l = -1;
            while((line = br.readLine()) != null){
                if(l < 0) {
                    String[] str = line.split(Tokens.SEP_OF_ELEMENTS);
                    try {
                        trainLayers = new TrainLayer[Integer.parseInt(str[0])];
                        for (int i = 0; i < trainLayers.length; i++) {
                            if (Integer.parseInt(str[i + 1]) <= 0) return 1;
                            trainLayers[i] = new TrainLayer(Integer.parseInt(str[i + 1]), (i < trainLayers.length - 1 ? Integer.parseInt(str[i + 2]) : 0));
                        }
                        l++;
                        continue;
                    } catch (NumberFormatException e) {
                        throw new RuntimeException(e);
                    }
                }
                if(l == trainLayers.length - 1) return 1;
                String[] w = line.split(Tokens.SEP_OF_OBJECTS);
                for (int j = 0; j < trainLayers[l].getLength(); j++) {
                    if (!trainLayers[l].setWeight(w[j].split(Tokens.SEP_OF_ELEMENTS), j)) return 1;
                }
                if (!trainLayers[l].setWeightB(w[trainLayers[l].getLength()].split(Tokens.SEP_OF_ELEMENTS))) return 1;
                l++;
            }
            layers = trainLayers;
            return 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int loadFromString(String content) {
        String[] lines = content.split(Tokens.SEP_OF_LAYERS);
        System.out.println(lines[0]);
        int l = -1;
        for(String line : lines){
            if(l < 0) {
                String[] str = line.split(Tokens.SEP_OF_ELEMENTS);
                try {
                    trainLayers = new TrainLayer[Integer.parseInt(str[0])];
                    for (int i = 0; i < trainLayers.length; i++) {
                        if (Integer.parseInt(str[i + 1]) <= 0) return 1;
                        trainLayers[i] = new TrainLayer(Integer.parseInt(str[i + 1]), (i < trainLayers.length - 1 ? Integer.parseInt(str[i + 2]) : 0));
                    }
                    l++;
                    continue;
                } catch (NumberFormatException e) {
                    throw new RuntimeException(e);
                }
            }
            if(l == trainLayers.length - 1) return 1;
            String[] w = line.split(Tokens.SEP_OF_OBJECTS);
            for (int j = 0; j < trainLayers[l].getLength(); j++) {
                if (!trainLayers[l].setWeight(w[j].split(Tokens.SEP_OF_ELEMENTS), j)) return 1;
            }
            if (!trainLayers[l].setWeightB(w[trainLayers[l].getLength()].split(Tokens.SEP_OF_ELEMENTS))) return 1;
            l++;
        }
        layers = trainLayers;
        return 0;
    }

    @Override
    public double[] getDeltaOfInputLayer() {
        return trainLayers[0].getDeltas();
    }
}
