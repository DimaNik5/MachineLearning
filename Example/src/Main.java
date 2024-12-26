
import com.ai.networks.Network;
import com.ai.networks.Teacher;
import com.ai.networks.Training;
import com.ai.networks.perceptron.Perceptron;
import com.ai.networks.perceptron.train.backpropagation.TrainPerceptronBackpropagation;
import com.ai.networks.perceptron.train.teacher.PerceptronTeacher;

/**
 * Пример работы внешней библиотеки Network.
 * В данном случае показан пример решения задачи исключающего ИЛИ
 */
public class Main {
    public static void main(String[] args) {
        String filename = System.getProperty("user.dir") + "/weight.txt";

        // Создание нового перцептрона
        Training<Double, Double> tp = new TrainPerceptronBackpropagation(new int[]{2, 8, 8, 1}, 0.03, 0.07, 10);
        // Загрузка перцептрона из файла
        //Training<Double, Double> tp = new TrainPerceptronBackpropagation(filename, 0.03, 0.07, 10);

        // Создание датасета
        Teacher<Double, Double> t = new PerceptronTeacher();
        t.setData(new Double[][]{
                        {0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}
                },
                new Double[][]{
                        {0.0}, {1.0}, {1.0}, {0.0}
                });

        // Обучение на протяжении 100000 эпох
        tp.training(t, 100000);

        /*
          Вывод состояние обучения.
          PrintStream - поток для вывода. При null создается по умолчанию поток с корректными русскими символами для IDE.
          Можно создать свой или использовать System.out
         */
        tp.printingStatus(null);

        try {
            // Ожидания конца обучения, которое происходит в параллельном потоке
            tp.getThread().join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        // Сохранение в файл
        tp.save(filename);

        // Создание рабочего перцептрона по обученным данным
        Network<Double, Double> p = new Perceptron();
        // Загрузка из файла
        p.loadFromFile(filename);

        // Установка входных значений
        p.setInput(new Double[]{0.0, 0.0});
        // Подсчет
        p.counting();
        // Получение выходных значений
        System.out.println("0 XOR 0 = " + p.getOutput()[0]);

        p.setInput(new Double[]{0.0, 1.0});
        p.counting();
        System.out.println("0 XOR 1 = " + p.getOutput()[0]);

        p.setInput(new Double[]{1.0, 0.0});
        p.counting();
        System.out.println("1 XOR 0 = " + p.getOutput()[0]);

        p.setInput(new Double[]{1.0, 1.0});
        p.counting();
        System.out.println("1 XOR 1 = " + p.getOutput()[0]);
    }
}