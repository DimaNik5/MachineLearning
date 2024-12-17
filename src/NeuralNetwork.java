import networks.Network;
import networks.Teacher;
import networks.perceptron.Perceptron;
import networks.perceptron.train.PerceptronTeacher;
import networks.perceptron.train.TrainPerceptron;

public class NeuralNetwork {
    public static void main(String[] args) throws InterruptedException {
        String filename = System.getProperty("user.dir") + "/weight.txt";
        TrainPerceptron tp = new TrainPerceptron(filename, 0.007, 0.003, 10);
        //TrainPerceptron tp = new TrainPerceptron(new int[]{2, 8, 8, 8, 1}, 0.07, 0.03, 10);
        Teacher t = new PerceptronTeacher();
        t.setData(new Double[][]{
                {0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}
                },
                new Double[][]{
                        {0.0}, {1.0}, {1.0}, {0.0}
        });

        tp.training(t, 100000);
        tp.printingStatus();
        tp.getThread().join();
        tp.save(filename);

    }
}