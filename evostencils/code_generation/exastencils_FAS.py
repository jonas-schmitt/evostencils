from evostencils.code_generation.layer4 import *


class ProgramGeneratorFAS:
    def __init__(self, solution, rhs, residual, FASApproximation,
                 restriction, prolongation, op_linear, op_nonlinear,
                 fct_name_mgcycle, max_level, min_level, fct_cgs, fct_smoother=None):

        # initialize max and min grid levels
        self.maxlevel = max_level
        self.minlevel = min_level

        # get function names if already defined in exa4
        self.fct_CGS = fct_cgs
        self.fct_smoother = fct_smoother

         # bool to clip multiple paths to the same 'rhs' node i.e. smoothing, restriction
        self.update_rhs = {}

        # init fields/stencils for each level
        self.solution = {}
        self.rhs = {}
        self.residual = {}
        self.FASApproximation = {}
        self.restriction = {}
        self.prolongation = {}
        self.op_linear = {}
        if op_nonlinear is not None:
            self.op_nonlinear = {}
        else:
            self.op_nonlinear = None

        for i in range(min_level, max_level + 1):
            self.solution[i] = Field(solution, i)
            self.rhs[i] = Field(rhs, i)
            self.residual[i] = Field(residual, i)
            self.FASApproximation[i] = Field(FASApproximation, i)
            self.restriction[i] = Field(restriction, i)
            self.prolongation[i] = Field(prolongation, i)
            self.op_linear[i] = Field(op_linear, i)
            self.update_rhs[i] = True
            if op_nonlinear is not None:
                self.op_nonlinear[i] = Field(op_nonlinear, i)

        # mgcycle generated
        self.fct_name = fct_name_mgcycle + f"@{max_level}"
        self.fct_mgcycle = None
        self.fct_body = []

    def traverse_graph(self, expression):
        def level(obj):
            return obj.grid[0].level

        def updateSolution():
            solution = self.traverse_graph(expression.approximation)
            correction = self.traverse_graph(expression.correction)

            if correction is not None:
                # update solution fields with correction
                self.fct_body.append(FieldLoop(solution, [Update(solution, correction)]))

                # apply bc
                self.fct_body.append(ApplyBC(solution))

                # communicate solution fields
                self.fct_body.append(Communicate(solution))

            return solution

        def updateFASApproximation():
            op_restriction = self.traverse_graph(expression.operand1)
            #solution = self.generate_multigrid(expression.operand2)

            # Store restricted solution in a separate field
            FASApproximation = self.FASApproximation[cur_lvl]
            solution_field = self.solution[cur_lvl]
            loop_stms = []
            loop_stms.append(Assignment(FASApproximation, Multiplication([op_restriction, solution_field])))
            loop_stms.append(Assignment(solution_field, FASApproximation))  # Initialize solution field at coarser level
            self.fct_body.append(FieldLoop(FASApproximation, loop_stms))

            # Communicate fields
            self.fct_body.append(Communicate(FASApproximation))
            self.fct_body.append(Communicate(solution_field))

            return FASApproximation

        def updateRHS(rhs_obj):
            if type(rhs_obj).__name__ != "RightHandSide":  # not the finest grid i.e cur_lvl < max_level
                assert cur_lvl < self.maxlevel

                # update rhs field
                if self.update_rhs[cur_lvl]: # rhs at the current level is not updated
                    rhs_expr = self.traverse_graph(rhs_obj)
                    rhs = self.rhs[cur_lvl]
                    self.fct_body.append(FieldLoop(rhs, [Assignment(rhs, rhs_expr)]))
                    self.update_rhs[cur_lvl] = False

        def updateResidual():
            solution = self.solution[cur_lvl]
            op_linear = self.traverse_graph(expression.operator)
            residual = self.residual[cur_lvl]
            rhs = self.rhs[cur_lvl]

            # Update residual fields
            loop_stms = []
            mul_expr = Multiplication([op_linear, solution])
            if self.op_nonlinear is not None:
                op_nonlinear = self.op_nonlinear[cur_lvl]
                mul_expr = Addition([mul_expr, Multiplication([op_nonlinear, solution])])
            loop_stms.append(Assignment(residual,
                                        Subtraction(rhs, mul_expr)))
            self.fct_body.append(FieldLoop(residual, loop_stms))

            # apply bc
            self.fct_body.append(ApplyBC(residual))
            
            # Communicate fields
            self.fct_body.append(Communicate(residual))

            return residual

        def updateFASerror():
            solution = self.traverse_graph(expression.operand1)
            FASApproximation = self.FASApproximation[cur_lvl]

            # subtract FAS approximation (restricted finer level solution) from solution at current level
            self.fct_body.append(FieldLoop(solution, [Update(solution, FASApproximation, "-=")]))

            # communicate solution field
            self.fct_body.append(Communicate(solution))

            return solution

        def solve():
            rhs_obj = expression.operand2

            # update rhs at coarse grid
            updateRHS(rhs_obj)

            # call coarse grid solver
            self.fct_body.append(FunctionCall(self.fct_CGS+ f'@{cur_lvl}'))

            return self.solution[cur_lvl]

        def smoothing():
            residual = expression.operand2
            updateRHS(residual.rhs)

            # call smoother
            self.fct_body.append(FunctionCall(self.fct_smoother + f'@{cur_lvl}'))

        cur_lvl = level(expression)
        expr_type = type(expression).__name__
        if expr_type == "Cycle":
            solution = updateSolution()
            return solution

        elif expr_type == "Multiplication":
            operator = expression.operand1
            operand = expression.operand2
            op_type = type(operator).__name__
            if op_type == "CoarseGridSolver":
                solution = solve()
                return solution
            elif op_type == "Inverse" and self.fct_smoother is not None:  # call predefined smoother
                smoothing()
                return None
            elif op_type == "Operator":
                operand = self.traverse_graph(operand)
                op_linear = self.traverse_graph(operator)
                mul_expr = Multiplication([op_linear, operand])
                if self.op_nonlinear is not None:
                    op_nonlinear = self.op_nonlinear[cur_lvl]
                    mul_expr = Addition([mul_expr, Multiplication([op_nonlinear, operand])])
                return mul_expr
            elif op_type == "Restriction" and ("Approximation" in type(operand).__name__ or type(operand).__name__ == "Cycle"):  # FAS expression
                FASApproximation = updateFASApproximation()
                return FASApproximation
            else:
                operand1 = self.traverse_graph(expression.operand1)
                operand2 = self.traverse_graph(expression.operand2)
                mul_expr = Multiplication([operand1, operand2])

                return mul_expr
        elif expr_type == "Addition":
            operand1 = self.traverse_graph(expression.operand1)
            operand2 = self.traverse_graph(expression.operand2)

            add_expr = Addition([operand1, operand2])

            return add_expr
        elif expr_type == "Subtraction":  # FAS expression
            error = updateFASerror()
            return error

        elif "Residual" in expr_type:
            updateRHS(expression.rhs)
            residual = updateResidual()
            return residual

        elif "Approximation" in expr_type:
            return self.solution[cur_lvl]

        elif expr_type == "RightHandSide":
            return self.rhs[cur_lvl]

        elif expr_type == "Prolongation":
            self.update_rhs[cur_lvl-1] = True # rhs at 'cur_lvl-1' needs to be updated
            return self.prolongation[cur_lvl - 1]

        elif expr_type == "Restriction":
            return self.restriction[cur_lvl + 1]

        elif expr_type == "Operator":
            return self.op_linear[cur_lvl]

    def generate_mgfunction(self, expression):
        self.traverse_graph(expression)
        self.fct_mgcycle = Function(self.fct_name, self.fct_body)
        return print_exa(self.fct_mgcycle)

