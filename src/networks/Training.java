package networks;

public interface Training{

    int training(Teacher dataset, int epochs);

    int printingStatus();

    void stop();

    Thread getThread();

    boolean isTraining();

    void save(String file);

    void save();

    double[] getErrors();
}
