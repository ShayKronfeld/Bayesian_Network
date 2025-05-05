import java.util.*;

/**
 * Represents a Bayesian Network.
 * <p>
 * This class manages a set of {@link Variable} objects, their dependencies (parent-child relationships),
 * and provides tools for traversal such as topological sorting.
 */
class BayesianNetwork {

    // --- Fields ---

    /** Maps variable names to their corresponding {@link Variable} objects. */
    private Map<String, Variable> variables = new HashMap<>();

    // --- Variable Management ---

    /**
     * Adds a variable to the network. If a variable with the same name already exists,
     * its outcomes will be merged (if any are missing).
     *
     * @param var the variable to add
     */
    public void addVariable(Variable var) {
        Variable existing = variables.get(var.getName());
        if (existing == null) {
            variables.put(var.getName(), var);
        } else {
            for (String outcome : var.getOutcomes()) {
                if (!existing.getOutcomes().contains(outcome)) {
                    existing.addOutcome(outcome);
                }
            }
        }
    }

    /**
     * Retrieves a variable by its name.
     *
     * @param name the name of the variable
     * @return the {@link Variable} object, or null if not found
     */
    public Variable getVariable(String name) {
        return variables.get(name);
    }

    /**
     * @return a collection of all variables in the network
     */
    public Collection<Variable> getVariables() {
        return variables.values();
    }

    /**
     * @return the internal map of variable names to {@link Variable} objects
     */
    public Map<String, Variable> getVariableMap() {
        return variables;
    }

    // --- Dependency Navigation ---

    /**
     * Retrieves the names of all children of a given variable.
     * A child is a variable that has this variable listed as a parent.
     *
     * @param varName the name of the variable whose children to find
     * @return a list of names of the children
     */
    public List<String> getChildren(String varName) {
        List<String> children = new ArrayList<>();
        for (Variable variable : variables.values()) {
            if (variable.getParents().contains(varName)) {
                children.add(variable.getName());
            }
        }
        return children;
    }

    // --- Topological Sorting ---

    /**
     * Returns a list of variables sorted in topological order.
     * Topological order ensures that every parent of a variable appears before the variable itself.
     *
     * @return a topologically sorted list of {@link Variable} objects
     */
    public List<Variable> getVariablesInTopologicalOrder() {
        List<Variable> ordered = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        for (Variable var : variables.values()) {
            dfs(var, visited, ordered);
        }
        return ordered;
    }

    /**
     * Helper method for depth-first search (DFS) used in topological sorting.
     *
     * @param var the current variable being visited
     * @param visited a set of already visited variable names
     * @param ordered the list being populated in topological order
     */
    private void dfs(Variable var, Set<String> visited, List<Variable> ordered) {
        if (visited.contains(var.getName())) return;
        visited.add(var.getName());

        for (String parentName : var.getParents()) {
            dfs(variables.get(parentName), visited, ordered);
        }

        ordered.add(var);
    }
}
