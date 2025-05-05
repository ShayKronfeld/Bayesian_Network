/**
 * Interface for Bayesian network inference algorithms.
 * Provides methods for tracking arithmetic operations during inference.
 */
public interface InferenceAlgorithm {

    /**
     * Increments the counter for multiplication operations.
     * Should be called whenever a multiplication is performed during inference.
     */
    void incrementMultiplications();

    /**
     * Increments the counter for addition operations.
     * Should be called whenever an addition is performed during inference.
     */
    void incrementAdditions();
}
