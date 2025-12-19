from pykokkos.interface.views import ViewType
from pykokkos.core import cppast

def upper_bound(view: ViewType, size: int, value) -> int:
    """
    Perform an upper bound search on a view
    
    Returns the index of the first element greater than value,
    similar to std::upper_bound or thrust::upper_bound.
    
    :param view: the view to search (must be sorted)
    :param size: the number of elements to search
    :param value: the value to search for
    :returns: the index of the first element greater than value
    """
    pass


def generate_upper_bound_binary_search(
    view_expr: cppast.Expr, size_expr: cppast.Expr, value_expr: cppast.Expr
) -> cppast.CompoundStmt:
    """
    Generate binary search implementation for upper_bound.
    Returns a CompoundStmt that implements:
    
    int left = -1;
    int right = size;
    int mid;
    while (left + 1 < right) {
        mid = left + ((right - left) >> 1);
        if (view[mid] > value) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
    """
    
    # Variable declarations
    int_type = cppast.PrimitiveType("int32_t")
    
    # int left = -1;
    left_var = cppast.DeclRefExpr("left")
    left_init = cppast.IntegerLiteral(-1)
    left_decl = cppast.VarDecl(int_type, left_var, left_init)
    left_stmt = cppast.DeclStmt(left_decl)
    
    # int right = size;
    right_var = cppast.DeclRefExpr("right")
    right_decl = cppast.VarDecl(int_type, right_var, size_expr)
    right_stmt = cppast.DeclStmt(right_decl)
    
    # int mid;
    mid_var = cppast.DeclRefExpr("mid")
    mid_decl = cppast.VarDecl(int_type, mid_var, None)
    mid_stmt = cppast.DeclStmt(mid_decl)
    
    # while condition: left + 1 < right
    left_plus_one = cppast.BinaryOperator(left_var, cppast.IntegerLiteral(1), cppast.BinaryOperatorKind.Add)
    while_cond = cppast.BinaryOperator(left_plus_one, right_var, cppast.BinaryOperatorKind.LT)
    
    # mid = left + ((right - left) >> 1);
    right_minus_left = cppast.BinaryOperator(right_var, left_var, cppast.BinaryOperatorKind.Sub)
    shift_expr = cppast.BinaryOperator(right_minus_left, cppast.IntegerLiteral(1), cppast.BinaryOperatorKind.Shr)
    mid_calc = cppast.BinaryOperator(left_var, shift_expr, cppast.BinaryOperatorKind.Add)
    mid_assign = cppast.AssignOperator([mid_var], mid_calc, cppast.BinaryOperatorKind.Assign)
    
    # if (view[mid] > value)
    view_ref = view_expr if isinstance(view_expr, cppast.DeclRefExpr) else cppast.DeclRefExpr("view")
    view_access = cppast.CallExpr(view_ref, [mid_var])
    if_cond = cppast.BinaryOperator(view_access, value_expr, cppast.BinaryOperatorKind.GT)
    
    # right = mid;
    right_assign = cppast.AssignOperator([right_var], mid_var, cppast.BinaryOperatorKind.Assign)
    
    # left = mid;
    left_assign = cppast.AssignOperator([left_var], mid_var, cppast.BinaryOperatorKind.Assign)
    
    # if-else statement
    if_stmt = cppast.IfStmt(if_cond, right_assign, left_assign)
    
    # while body
    while_body = cppast.CompoundStmt([mid_assign, if_stmt])
    while_stmt = cppast.WhileStmt(while_cond, while_body)
    
    # return right;
    return_stmt = cppast.ReturnStmt(right_var)
    
    # Complete function body
    return cppast.CompoundStmt([left_stmt, right_stmt, mid_stmt, while_stmt, return_stmt])
