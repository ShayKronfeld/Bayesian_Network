import java.util.*;

/**
 * Represents a variable (node) in a Bayesian Network.
 * A variable has:
 * - a name
 * - a list of possible outcomes (values)
 * - a list of parent variables (by name)
 * - a Conditional Probability Table (CPT) stored as a flat array.
 */
class Variable {

    // --- Fields ---
    private String name;
    private List<String> outcomes = new ArrayList<>();
    private List<String> parents = new ArrayList<>();
    private double[] cpt; // Conditional Probability Table

    // --- Constructor ---

    /**
     * Creates a new variable with the specified name.
     * @param name the name of the variable.
     */
    public Variable(String name) {
        this.name = name;
    }

    // --- Getters and Setters ---

    /**
     * @return the name of the variable.
     */
    public String getName() {
        return name;
    }

    /**
     * @return the list of possible outcomes (values) for the variable.
     */
    public List<String> getOutcomes() {
        return outcomes;
    }

    /**
     * @return the list of parent variable names.
     */
    public List<String> getParents() {
        return parents;
    }

    /**
     * @return the Conditional Probability Table (CPT) as a flat array.
     */
    public double[] getCPT() {
        return cpt;
    }

    /**
     * Sets the CPT (Conditional Probability Table).
     * @param cpt a flat array representing the CPT.
     */
    public void setCPT(double[] cpt) {
        this.cpt = cpt;
    }

    // --- Modifiers ---

    /**
     * Adds a new outcome to the list of possible outcomes for this variable.
     * @param outcome the outcome to add.
     */
    public void addOutcome(String outcome) {
        outcomes.add(outcome);
    }

    /**
     * Adds a new parent variable by name.
     * @param parent the name of the parent variable.
     */
    public void addParent(String parent) {
        parents.add(parent);
    }

    // --- CPT Logic ---

    /**
     * Finds the index of a given outcome value in the outcomes list.
     * @param value the outcome value (e.g., "T", "F").
     * @return the index of the value in the outcomes list.
     * @throws IllegalArgumentException if the value is not found.
     */
    public int getOutcomeIndex(String value) {
        value = value.trim();
        for (int i = 0; i < outcomes.size(); i++) {
            if (outcomes.get(i).trim().equals(value)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Value '" + value + "' not found in variable '" + name + "'. Outcomes = " + outcomes);
    }

    /**
     * Retrieves the probability of this variable having a specific value
     * given a full assignment of the network (including all parents).
     *
     * CPT index is calculated such that:
     * - Parent variables are ordered from last to first (rightmost changes fastest)
     * - The variable itself is the fastest changing within the block
     *
     * @param assignment a complete assignment of variables to their values.
     * @param networkVariables the map of all variables in the network.
     * @return the probability according to the CPT.
     */
    public double getProbability(Map<String, String> assignment, Map<String, Variable> networkVariables) {
        int index = 0;
        int multiplier = 1;

        // Parents: from last to first
        for (int i = parents.size() - 1; i >= 0; i--) {
            String parentName = parents.get(i);
            Variable parent = networkVariables.get(parentName);
            String parentValue = assignment.get(parentName).trim();
            int parentIndex = parent.getOutcomeIndex(parentValue);
            index += parentIndex * multiplier;
            multiplier *= parent.getOutcomes().size();
        }

        // The variable itself
        String myValue = assignment.get(name).trim();
        int myValueIndex = getOutcomeIndex(myValue);
        index = index * outcomes.size() + myValueIndex;

        return cpt[index];
    }

    /**
     * Returns the list of parent values from the assignment,
     * ordered according to the parent's order in this variable.
     * @param assignment a complete assignment of variables to values.
     * @return ordered list of parent values.
     * @throws IllegalArgumentException if a parent is missing from the assignment.
     */
    public List<String> getOrderedParentValues(Map<String, String> assignment) {
        List<String> ordered = new ArrayList<>();
        for (String parent : parents) {
            String value = assignment.get(parent);
            if (value == null) {
                throw new IllegalArgumentException("Missing value for parent: " + parent);
            }
            ordered.add(value.trim());
        }
        return ordered;
    }
}
