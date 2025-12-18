from pykokkos.interface.views import ViewType
from pykokkos.core import cppast

def lower_bound(view: ViewType, size: int, value) -> int:
    """
    Perform a lower bound search on a view
    
    Returns the index of the first element not less than (i.e. greater or equal to) value,
    similar to std::lower_bound or thrust::lower_bound.
    
    Supported types: All orderable numeric types (int8, int16, int32, int64,
    uint8, uint16, uint32, uint64, float, double). Complex types are not
    supported as they cannot be ordered.
    
    :param view: the view to search (must be sorted in ascending order)
    :param size: the number of elements to search
    :param value: the value to search for (must match view's element type)
    :returns: the index of the first element >= value
    """
    pass


def generate_lower_bound_binary_search(
    view_expr: cppast.Expr, size_expr: cppast.Expr, value_expr: cppast.Expr
) -> cppast.CompoundStmt:
    """
    Generate binary search implementation for lower_bound.
    Returns a CompoundStmt that implements:
    
    int left = 0;
    int right = size;
    int mid;
    while (left < right) {
        mid = left + (right - left) / 2;
        if (view[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
    """
    
    # Variable declarations
    int_type = cppast.PrimitiveType("int32_t")
    
    # int left = 0;
    left_var = cppast.DeclRefExpr("left")
    left_init = cppast.IntegerLiteral(0)
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
    
    # while (left < right)
    while_cond = cppast.BinaryOperator(left_var, right_var, cppast.BinaryOperatorKind.LT)
    
    # mid = left + (right - left) / 2;
    right_minus_left = cppast.BinaryOperator(right_var, left_var, cppast.BinaryOperatorKind.Sub)
    div_expr = cppast.BinaryOperator(right_minus_left, cppast.IntegerLiteral(2), cppast.BinaryOperatorKind.Div)
    mid_calc = cppast.BinaryOperator(left_var, div_expr, cppast.BinaryOperatorKind.Add)
    mid_assign = cppast.AssignOperator([mid_var], mid_calc, cppast.BinaryOperatorKind.Assign)
    
    # if (view(mid) < value)
    view_ref = view_expr if isinstance(view_expr, cppast.DeclRefExpr) else cppast.DeclRefExpr("view")
    view_access = cppast.CallExpr(view_ref, [mid_var])
    if_cond = cppast.BinaryOperator(view_access, value_expr, cppast.BinaryOperatorKind.LT)
    
    # left = mid + 1;
    mid_plus_one = cppast.BinaryOperator(mid_var, cppast.IntegerLiteral(1), cppast.BinaryOperatorKind.Add)
    left_assign = cppast.AssignOperator([left_var], mid_plus_one, cppast.BinaryOperatorKind.Assign)
    
    # right = mid;
    right_assign = cppast.AssignOperator([right_var], mid_var, cppast.BinaryOperatorKind.Assign)
    
    # if-else statement
    if_stmt = cppast.IfStmt(if_cond, left_assign, right_assign)
    
    # while body
    while_body = cppast.CompoundStmt([mid_assign, if_stmt])
    while_stmt = cppast.WhileStmt(while_cond, while_body)
    
    # return left;
    return_stmt = cppast.ReturnStmt(left_var)
    
    # Complete function body
    return cppast.CompoundStmt([left_stmt, right_stmt, mid_stmt, while_stmt, return_stmt])
