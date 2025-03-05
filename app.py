from flask import Flask, request, jsonify
import gurobipy as gp
from gurobipy import GRB

app = Flask(__name__)

@app.route('/')
def home():
    return "Gurobi Solver is running!"

@app.route('/solve', methods=['POST'])
def solve():
    try:
        # Get data from request (expected as JSON)
        data = request.get_json()
        
        # Example: Extract values (modify as per your optimization model)
        a = data.get("a", 1)
        b = data.get("b", 1)

        # Create Gurobi model
        model = gp.Model("example")
        x = model.addVar(name="x", vtype=GRB.CONTINUOUS)
        model.setObjective(a * x, GRB.MAXIMIZE)
        model.addConstr(x <= b, "constraint")
        model.optimize()

        # Extract results
        solution = {"x": x.X, "objective": model.objVal}

        return jsonify(solution)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
