//===- BullsEyeAnalysis.cc - Analysis on MLIR code by BullsEye -------------------===//
//
// This file implements the analysis pass on MLIR using BullsEye.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/BullsEyeAnalysis.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "bullseye/bullseyelib.h"  

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/aff.h>
#include <isl/val.h>
#include <isl/union_set.h>
#include <isl/space.h>  
#include <isl/vertices.h>
#include <isl/constraint.h>
// #include <isl/isl-noexception.h>
#include <ostream>
#include <pet.h>
#include <cloog/cloog.h> 
#include <osl/util.h>
#include <osl/body.h>
#include <osl/extensions/extbody.h>
#include <osl/statement.h>
#include <osl/scop.h>
#include <osl/macros.h>
#include <osl/int.h>
#include <osl/util.h>
#include <osl/vector.h>
#include <osl/strings.h>
#include <osl/names.h>
#include <osl/relation.h>
#include <osl/extensions/arrays.h> 
#include <fstream> 
#include <cstdarg>
#include <string>
#include <vector>
#include <sstream>
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "cache-analysis"
#define OSL_SUPPORT 1

namespace std {
  template<> struct hash<mlir::Value> {
    std::size_t operator()(const mlir::Value& val) const noexcept {
      return hash<void*>()(val.getAsOpaquePointer());
    }
  };
}

namespace std {
std::ostream &operator<<(std::ostream &os, const std::vector<long> &vec) {
  for (int i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i < vec.size() - 1)
      os << " ";
  }
  return os;
}
} // namespace std
 

void rearrangeString(const char *input, char *output) {
    // Initialize variables to store the parts of the rearranged string
    char prefix[100];
    char middle[100];
    char postfix[100];

    // Initialize indices for the prefix, middle, and postfix parts
    int prefixIndex = 0;
    int middleIndex = 0;
    int postfixIndex = 0;

    // Initialize a flag to track if we are inside square brackets
    int inBracket = 0;

    // Initialize a flag to track if we should subtract 1 from the integer
    int subtractOne = 0;

    // Iterate through the input string
    for (int i = 0; input[i] != '\0'; i++) {
        if (input[i] == '[') {
            inBracket = 1;
            middle[middleIndex++] = '['; // Include the opening bracket in the middle
            subtractOne = 0; // Reset the subtractOne flag when entering a new set of square brackets
        } else if (input[i] == ']') {
            inBracket = 0;
            middle[middleIndex++] = ']';
            middle[middleIndex] = '\0'; // Null-terminate the middle part
        } else if (!inBracket) {
            prefix[prefixIndex++] = input[i];
            prefix[prefixIndex] = '\0'; // Null-terminate the prefix part
        } else {
            middle[middleIndex++] = input[i];
            if (input[i] == 'i' && isdigit(input[i + 1])) {
                // Subtract 1 from the integer immediately following 'i'
                middle[middleIndex++] = input[i + 1] - '1' + '0';
                i++; // Skip the integer to avoid processing it again
                subtractOne = 1; // Set the subtractOne flag
            } else if (subtractOne) {
                // Skip the integer if it was already subtracted from
                i++;
                subtractOne = 0;
            }
        }
    }

    // Null-terminate the postfix part
    postfix[postfixIndex] = '\0';

    // Combine the parts to form the rearranged string with the modified middle
    sprintf(output, "%s%s%s", prefix, middle, postfix);
}

std::unordered_map<std::string, unsigned> arraySizes; 

void generateDeclarations(std::vector<std::string> &reads, std::vector<std::string> &writes) {
    for (const std::string& read: reads) {
        // Find the position of the first '[' character
        size_t bracketPos = read.find('[');

        // Extract the substring before the '[' character
        std::string result = read.substr(0, bracketPos);

        char target = '['; // Character to count
        int count = 0;

        for (char c : read) {
            if (c == target) {
                count++;
            }
        }

        arraySizes[result] = count;
    }
}

std::string generateStatement(std::vector<std::string> &reads, std::vector<std::string> &writes, std::string statementText) {
    statementText = writes[0];
    statementText += " = ";
    for (const std::string& read : reads) { 
        statementText += read;
        if (read != reads.back()) {
            statementText += " + "; // Add a "+" sign between read operations
        }
    }
    statementText += ";";
    fprintf(stdout, "%s", statementText.c_str());

    return statementText;
}
 
/* Use the CLooG library to output a SCoP from OpenScop to C */
void print_scop_to_c(FILE* output, osl_scop_p scop) {
  CloogState* state;
  CloogOptions* options;
  CloogInput* input;
  struct clast_stmt* clast;

  state = cloog_state_malloc();
  options = cloog_options_malloc(state);
  options->openscop = 1;
  cloog_options_copy_from_osl_scop(scop, options);
  input = cloog_input_from_osl_scop(options->state, scop);
  clast = cloog_clast_create_from_input(input, options);
  clast_pprint(output, clast, 0, options);
  
  cloog_clast_free(clast);
  options->scop = NULL; // don't free the scop
  cloog_options_free(options);
  cloog_state_free(state); // the input is freed inside
}


static void pprint_name(FILE *dst, struct clast_name *n);
static void pprint_term(struct cloogoptions *i, FILE *dst, struct clast_term *t);
static void pprint_sum(struct cloogoptions *opt,
			FILE *dst, struct clast_reduction *r);
static void pprint_binary(struct cloogoptions *i,
			FILE *dst, struct clast_binary *b);
static void pprint_minmax_f(struct cloogoptions *info,
			FILE *dst, struct clast_reduction *r);
static void pprint_minmax_c(struct cloogoptions *info,
			FILE *dst, struct clast_reduction *r);
static void pprint_reduction(struct cloogoptions *i,
			FILE *dst, struct clast_reduction *r);
static void pprint_expr(struct cloogoptions *i, FILE *dst, struct clast_expr *e);
static void pprint_equation(struct cloogoptions *i,
			FILE *dst, struct clast_equation *eq);
static void pprint_assignment(struct cloogoptions *i, FILE *dst, 
			struct clast_assignment *a);
static void pprint_user_stmt(struct cloogoptions *options, FILE *dst,
		       struct clast_user_stmt *u);
static void pprint_guard(struct cloogoptions *options, FILE *dst, int indent,
		   struct clast_guard *g);
static void pprint_for(struct cloogoptions *options, FILE *dst, int indent,
		 struct clast_for *f);
static void pprint_stmt_list(struct cloogoptions *options, FILE *dst, int indent,
		       struct clast_stmt *s);


void pprint_name(FILE *dst, struct clast_name *n)
{
    fprintf(dst, "%s", n->name);
}

/**
 * This function returns a string containing the printing of a value (possibly
 * an iterator or a parameter with its coefficient or a constant).
 * - val is the coefficient or constant value,
 * - name is a string containing the name of the iterator or of the parameter,
 */
void pprint_term(struct cloogoptions *i, FILE *dst, struct clast_term *t)
{
    if (t->var) {
	int group = t->var->type == clast_expr_red &&
		    ((struct clast_reduction*) t->var)->n > 1;
	if (cloog_int_is_one(t->val))
	    ;
	else if (cloog_int_is_neg_one(t->val))
	    fprintf(dst, "-");
        else {
	    cloog_int_print(dst, t->val);
	    fprintf(dst, "*");
	}
	if (group)
	    fprintf(dst, "(");
	pprint_expr(i, dst, t->var);
	if (group)
	    fprintf(dst, ")");
    } else
	cloog_int_print(dst, t->val);
}

void pprint_sum(struct cloogoptions *opt, FILE *dst, struct clast_reduction *r)
{
    int i;
    struct clast_term *t;

    assert(r->n >= 1);
    assert(r->elts[0]->type == clast_expr_term);
    t = (struct clast_term *) r->elts[0];
    pprint_term(opt, dst, t);

    for (i = 1; i < r->n; ++i) {
	assert(r->elts[i]->type == clast_expr_term);
	t = (struct clast_term *) r->elts[i];
	if (cloog_int_is_pos(t->val))
	    fprintf(dst, "+");
	pprint_term(opt, dst, t);
    }
}

void pprint_binary(struct cloogoptions *i, FILE *dst, struct clast_binary *b)
{
    const char *s1 = NULL, *s2 = NULL, *s3 = NULL;
    int group = b->LHS->type == clast_expr_red &&
		((struct clast_reduction*) b->LHS)->n > 1;
    if (i->language == CLOOG_LANGUAGE_FORTRAN) {
	switch (b->type) {
	case clast_bin_fdiv:
	    s1 = "FLOOR(REAL(", s2 = ")/REAL(", s3 = "))";
	    break;
	case clast_bin_cdiv:
	    s1 = "CEILING(REAL(", s2 = ")/REAL(", s3 = "))";
	    break;
	case clast_bin_div:
	    if (group)
		s1 = "(", s2 = ")/", s3 = "";
	    else
		s1 = "", s2 = "/", s3 = "";
	    break;
	case clast_bin_mod:
	    s1 = "MOD(", s2 = ", ", s3 = ")";
	    break;
	}
    } else {
	switch (b->type) {
	case clast_bin_fdiv:
	    s1 = "floord(", s2 = ",", s3 = ")";
	    break;
	case clast_bin_cdiv:
	    s1 = "ceild(", s2 = ",", s3 = ")";
	    break;
	case clast_bin_div:
	    if (group)
		s1 = "(", s2 = ")/", s3 = "";
	    else
		s1 = "", s2 = "/", s3 = "";
	    break;
	case clast_bin_mod:
	    if (group)
		s1 = "(", s2 = ")%", s3 = "";
	    else
		s1 = "", s2 = "%", s3 = "";
	    break;
	}
    }
    fprintf(dst, "%s", s1);
    pprint_expr(i, dst, b->LHS);
    fprintf(dst, "%s", s2);
    cloog_int_print(dst, b->RHS);
    fprintf(dst, "%s", s3);
}

void pprint_minmax_f(struct cloogoptions *info, FILE *dst, struct clast_reduction *r)
{
    int i;
    if (r->n == 0)
	return;
    fprintf(dst, r->type == clast_red_max ? "MAX(" : "MIN(");
    pprint_expr(info, dst, r->elts[0]);
    for (i = 1; i < r->n; ++i) {
	fprintf(dst, ",");
	pprint_expr(info, dst, r->elts[i]);
    }
    fprintf(dst, ")");
}

void pprint_minmax_c(struct cloogoptions *info, FILE *dst, struct clast_reduction *r)
{
    int i;
    for (i = 1; i < r->n; ++i)
	fprintf(dst, r->type == clast_red_max ? "max(" : "min(");
    if (r->n > 0)
	pprint_expr(info, dst, r->elts[0]);
    for (i = 1; i < r->n; ++i) {
	fprintf(dst, ",");
	pprint_expr(info, dst, r->elts[i]);
	fprintf(dst, ")");
    }
}

void pprint_reduction(struct cloogoptions *i, FILE *dst, struct clast_reduction *r)
{
    switch (r->type) {
    case clast_red_sum:
	pprint_sum(i, dst, r);
	break;
    case clast_red_min:
    case clast_red_max:
	if (r->n == 1) {
	    pprint_expr(i, dst, r->elts[0]);
	    break;
	}
	if (i->language == CLOOG_LANGUAGE_FORTRAN)
	    pprint_minmax_f(i, dst, r);
	else
	    pprint_minmax_c(i, dst, r);
	break;
    default:
	assert(0);
    }
}

void pprint_expr(struct cloogoptions *i, FILE *dst, struct clast_expr *e)
{
    if (!e)
	return;
    switch (e->type) {
    case clast_expr_name:
	pprint_name(dst, (struct clast_name*) e);
	break;
    case clast_expr_term:
	pprint_term(i, dst, (struct clast_term*) e);
	break;
    case clast_expr_red:
	pprint_reduction(i, dst, (struct clast_reduction*) e);
	break;
    case clast_expr_bin:
	pprint_binary(i, dst, (struct clast_binary*) e);
	break;
    default:
	assert(0);
    }
}

void pprint_equation(struct cloogoptions *i, FILE *dst, struct clast_equation *eq)
{
    pprint_expr(i, dst, eq->LHS);
    if (eq->sign == 0)
	fprintf(dst, " == ");
    else if (eq->sign > 0)
	fprintf(dst, " >= ");
    else
	fprintf(dst, " <= ");
    pprint_expr(i, dst, eq->RHS);
}

void pprint_assignment(struct cloogoptions *i, FILE *dst, 
			struct clast_assignment *a)
{
    if (a->LHS)
	fprintf(dst, "%s = ", a->LHS);
    pprint_expr(i, dst, a->RHS);
}


/**
 * pprint_osl_body function:
 * this function pretty-prints the OpenScop body of a given statement.
 * It returns 1 if it succeeds to find an OpenScop body to print for
 * that statement, 0 otherwise.
 * \param[in] options CLooG Options.
 * \param[in] dst     Output stream.
 * \param[in] u       Statement to print the OpenScop body.
 * \return 1 on success to pretty-print an OpenScop body for u, 0 otherwise.
 */
int pprint_osl_body(struct cloogoptions *options, FILE *dst,
                    struct clast_user_stmt *u) {
#ifdef OSL_SUPPORT
  int i;
  char *expr, *tmp;
  struct clast_stmt *t;
  osl_scop_p scop = options->scop;
  osl_statement_p stmt;
  osl_body_p body;

  if ((scop != NULL) &&
      (osl_statement_number(scop->statement) >= u->statement->number)) {
    stmt = scop->statement;

    /* Go to the convenient statement in the SCoP. */
    for (i = 1; i < u->statement->number; i++)
      stmt = stmt->next;

    /* Ensure it has a printable body. */
    body = osl_statement_get_body(stmt);
    if ((body != NULL) &&
        (body->expression != NULL) &&
        (body->iterators != NULL)) {
      expr = osl_util_identifier_substitution(body->expression->string[0],
                                              body->iterators->string);
      tmp = expr;
      /* Print the body expression, substituting the @...@ markers. */
      while (*expr) {
        if (*expr == '@') {
          int iterator;
          expr += sscanf(expr, "@%d", &iterator) + 2; /* 2 for the @s */
          t = u->substitutions;
          for (i = 0; i < iterator; i++)
            t = t->next;
          pprint_assignment(options, dst, (struct clast_assignment *)t);
        } else {
          fprintf(dst, "%c", *expr++);
        }
      }
      fprintf(dst, "\n");
      free(tmp);
      return 1;
    }
  }
#endif
  return 0;
}

/* pprint_parentheses_are_safer function:
 * this function returns 1 if it decides that it would be safer to put
 * parentheses around the clast_assignment when it is used as a macro
 * parameter, 0 otherwise.
 * \param[in] s Pointer to the clast_assignment to check.
 * \return 1 if we should print parentheses around s, 0 otherwise.
 */
static int pprint_parentheses_are_safer(struct clast_assignment * s) {
  /* Expressions of the form X = Y should not be used in macros, so we
   * consider readability first for them and avoid parentheses.
   * Also, expressions having only one term can live without parentheses.
   */
  if ((s->LHS) ||
      (s->RHS->type == clast_expr_term) ||
      ((s->RHS->type == clast_expr_red) &&
       (((struct clast_reduction *)(s->RHS))->n == 1) &&
       (((struct clast_reduction *)(s->RHS))->elts[0]->type ==
        clast_expr_term)))
    return 0;

  return 1;
}

void pprint_user_stmt(struct cloogoptions *options, FILE *dst,
		       struct clast_user_stmt *u)
{
    int parenthesis_to_close = 0;
    struct clast_stmt *t;

    if (pprint_osl_body(options, dst, u))
      return;
    
    if (u->statement->name)
	fprintf(dst, "%s", u->statement->name);
    else
	fprintf(dst, "S%d", u->statement->number);
    fprintf(dst, "(");
    for (t = u->substitutions; t; t = t->next) {
	assert(CLAST_STMT_IS_A(t, stmt_ass));
        if (pprint_parentheses_are_safer((struct clast_assignment *)t)) {
	  fprintf(dst, "(");
          parenthesis_to_close = 1;
        }
	pprint_assignment(options, dst, (struct clast_assignment *)t);
	if (t->next) {
            if (parenthesis_to_close) {
	      fprintf(dst, ")");
              parenthesis_to_close = 0;
            }
	    fprintf(dst, ",");
        }
    }
    if (parenthesis_to_close)
      fprintf(dst, ")");
    fprintf(dst, ")");
    if (options->language != CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst, ";");
    fprintf(dst, "\n");
}

void pprint_guard(struct cloogoptions *options, FILE *dst, int indent,
		   struct clast_guard *g)
{
    int k;
    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst,"IF ");
    else
	fprintf(dst,"if ");
    if (g->n > 1)
	fprintf(dst,"(");
    for (k = 0; k < g->n; ++k) {
	if (k > 0) {
	    if (options->language == CLOOG_LANGUAGE_FORTRAN)
		fprintf(dst," .AND. ");
	    else
		fprintf(dst," && ");
	}
	fprintf(dst,"(");
        pprint_equation(options, dst, &g->eq[k]);
	fprintf(dst,")");
    }
    if (g->n > 1)
	fprintf(dst,")");
    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst," THEN\n");
    else
	fprintf(dst," {\n");

    pprint_stmt_list(options, dst, indent + INDENT_STEP, g->then);

    fprintf(dst, "%*s", indent, "");
    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst,"END IF\n"); 
    else
	fprintf(dst,"}\n"); 
}

void pprint_for(struct cloogoptions *options, FILE *dst, int indent,
		 struct clast_for *f)
{
    if (options->language == CLOOG_LANGUAGE_C) {
        if (f->time_var_name) {
            fprintf(dst, "IF_TIME(%s_start = cloog_util_rtclock());\n",
                    (f->time_var_name) ? f->time_var_name : "");
        }
        if ((f->parallel & CLAST_PARALLEL_OMP) && !(f->parallel & CLAST_PARALLEL_MPI)) {
            if (f->LB) {
                fprintf(dst, "lbp=");
                pprint_expr(options, dst, f->LB);
                fprintf(dst, ";\n");
            }
            if (f->UB) {
                fprintf(dst, "%*s", indent, "");
                fprintf(dst, "ubp=");
                pprint_expr(options, dst, f->UB);
                fprintf(dst, ";\n");
            }
            fprintf(dst, "#pragma omp parallel for%s%s%s%s%s%s\n",
                    (f->private_vars)? " private(":"",
                    (f->private_vars)? f->private_vars: "",
                    (f->private_vars)? ")":"",
                    (f->reduction_vars)? " reduction(": "",
                    (f->reduction_vars)? f->reduction_vars: "",
                    (f->reduction_vars)? ")": "");
            fprintf(dst, "%*s", indent, "");
        }
        if ((f->parallel & CLAST_PARALLEL_VEC) && !(f->parallel & CLAST_PARALLEL_OMP)
               && !(f->parallel & CLAST_PARALLEL_MPI)) {
            if (f->LB) {
                fprintf(dst, "lbv=");
                pprint_expr(options, dst, f->LB);
                fprintf(dst, ";\n");
            }
            if (f->UB) {
                fprintf(dst, "%*s", indent, "");
                fprintf(dst, "ubv=");
                pprint_expr(options, dst, f->UB);
                fprintf(dst, ";\n");
            }
            fprintf(dst, "%*s#pragma ivdep\n", indent, "");
            fprintf(dst, "%*s#pragma vector always\n", indent, "");
            fprintf(dst, "%*s", indent, "");
        }
        if (f->parallel & CLAST_PARALLEL_MPI) {
            if (f->LB) {
                fprintf(dst, "_lb_dist=");
                pprint_expr(options, dst, f->LB);
                fprintf(dst, ";\n");
            }
            if (f->UB) {
                fprintf(dst, "%*s", indent, "");
                fprintf(dst, "_ub_dist=");
                pprint_expr(options, dst, f->UB);
                fprintf(dst, ";\n");
            }
            fprintf(dst, "%*s", indent, "");
            fprintf(dst, "polyrt_loop_dist(_lb_dist, _ub_dist, nprocs, my_rank, &lbp, &ubp);\n");
            if (f->parallel & CLAST_PARALLEL_OMP) {
                fprintf(dst, "#pragma omp parallel for%s%s%s%s%s%s\n",
                        (f->private_vars)? " private(":"",
                        (f->private_vars)? f->private_vars: "",
                        (f->private_vars)? ")":"",
                        (f->reduction_vars)? " reduction(": "",
                        (f->reduction_vars)? f->reduction_vars: "",
                        (f->reduction_vars)? ")": "");
            }
            fprintf(dst, "%*s", indent, "");
        }

    }

    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst, "DO ");
    else
	fprintf(dst, "for (");

    if (f->LB) {
	fprintf(dst, "%s=", f->iterator);
        if (f->parallel & (CLAST_PARALLEL_OMP | CLAST_PARALLEL_MPI)) {
            fprintf(dst, "lbp");
        }else if (f->parallel & CLAST_PARALLEL_VEC){
            fprintf(dst, "lbv");
        }else{
	pprint_expr(options, dst, f->LB);
        }
    } else if (options->language == CLOOG_LANGUAGE_FORTRAN)
	cloog_die("unbounded loops not allowed in FORTRAN.\n");

    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst,", ");
    else
	fprintf(dst,";");

    if (f->UB) { 
	if (options->language != CLOOG_LANGUAGE_FORTRAN)
	    fprintf(dst,"%s<=", f->iterator);

        if (f->parallel & (CLAST_PARALLEL_OMP | CLAST_PARALLEL_MPI)) {
            fprintf(dst, "ubp");
        }else if (f->parallel & CLAST_PARALLEL_VEC){
            fprintf(dst, "ubv");
        }else{
            pprint_expr(options, dst, f->UB);
        }
    }else if (options->language == CLOOG_LANGUAGE_FORTRAN)
	cloog_die("unbounded loops not allowed in FORTRAN.\n");

    if (options->language == CLOOG_LANGUAGE_FORTRAN) {
	if (cloog_int_gt_si(f->stride, 1))
	    cloog_int_print(dst, f->stride);
	fprintf(dst,"\n");
    }
    else {
	if (cloog_int_gt_si(f->stride, 1)) {
	    fprintf(dst,";%s+=", f->iterator);
	    cloog_int_print(dst, f->stride);
	    fprintf(dst, ") {\n");
      } else
	fprintf(dst, ";%s++) {\n", f->iterator);
    }

    pprint_stmt_list(options, dst, indent + INDENT_STEP, f->body);

    fprintf(dst, "%*s", indent, "");
    if (options->language == CLOOG_LANGUAGE_FORTRAN)
	fprintf(dst,"END DO\n") ; 
    else
	fprintf(dst,"}\n") ; 

    if (options->language == CLOOG_LANGUAGE_C) {
        if (f->time_var_name) {
            fprintf(dst, "IF_TIME(%s += cloog_util_rtclock() - %s_start);\n",
                    (f->time_var_name) ? f->time_var_name : "",
                    (f->time_var_name) ? f->time_var_name : "");
        }
    }
}

void pprint_stmt_list(struct cloogoptions *options, FILE *dst, int indent,
		       struct clast_stmt *s)
{
    for ( ; s; s = s->next) {
	if (CLAST_STMT_IS_A(s, stmt_root))
	    continue;
	fprintf(dst, "%*s", indent, "");
	if (CLAST_STMT_IS_A(s, stmt_ass)) {
	    pprint_assignment(options, dst, (struct clast_assignment *) s);
	    if (options->language != CLOOG_LANGUAGE_FORTRAN)
		fprintf(dst, ";");
	    fprintf(dst, "\n");
	} else if (CLAST_STMT_IS_A(s, stmt_user)) {
	    pprint_user_stmt(options, dst, (struct clast_user_stmt *) s);
	} else if (CLAST_STMT_IS_A(s, stmt_for)) {
	    pprint_for(options, dst, indent, (struct clast_for *) s);
	} else if (CLAST_STMT_IS_A(s, stmt_guard)) {
	    pprint_guard(options, dst, indent, (struct clast_guard *) s);
	} else if (CLAST_STMT_IS_A(s, stmt_block)) {
	    fprintf(dst, "{\n");
	    pprint_stmt_list(options, dst, indent + INDENT_STEP, 
				((struct clast_block *)s)->body);
	    fprintf(dst, "%*s", indent, "");
	    fprintf(dst, "}\n");
	} else {
	    assert(0);
	}
    }
}


/******************************************************************************
 *                       Pretty Printing (dirty) functions                    *
 ******************************************************************************/

void clast_pprint(FILE *foo, struct clast_stmt *root,
		  int indent, CloogOptions *options)
{
    pprint_stmt_list(options, foo, indent, root);
}


void clast_pprint_expr(struct cloogoptions *i, FILE *dst, struct clast_expr *e)
{
    pprint_expr(i, dst, e);
}

/**
 * osl_relation_sprint_type function:
 * this function prints the textual type of an osl_relation_t structure into
 * a string, according to the OpenScop specification, and returns that string.
 * \param[in] relation The relation whose type has to be printed.
 * \return A string containing the relation type.
 */
static
char * osl_relation_sprint_type(osl_relation_p relation) {
  char * string = NULL;
  
  OSL_malloc(string, char *, OSL_MAX_STRING * sizeof(char));
  string[0] = '\0';

  if (relation != NULL) {
    switch (relation->type) {
      case OSL_UNDEFINED: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_UNDEFINED);
        break;
      }
      case OSL_TYPE_CONTEXT: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_CONTEXT);
        break;
      }
      case OSL_TYPE_DOMAIN: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_DOMAIN);
        break;
      }
      case OSL_TYPE_SCATTERING: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_SCATTERING);
        break;
      }
      case OSL_TYPE_READ: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_READ);
        break;
      }
      case OSL_TYPE_WRITE: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_WRITE);
        break;
      }
      case OSL_TYPE_MAY_WRITE: {
        snprintf(string, OSL_MAX_STRING, OSL_STRING_MAY_WRITE);
        break;
      }
      default: {
        OSL_warning("unknown relation type, replaced with " OSL_STRING_UNDEFINED);
        snprintf(string, OSL_MAX_STRING, OSL_STRING_UNDEFINED);
      }
    }
  }

  return string;
}



/**
 * osl_relation_strings function:
 * this function creates a NULL-terminated array of strings from an
 * osl_names_t structure in such a way that the ith string is the "name"
 * corresponding to the ith column of the constraint matrix.
 * \param[in] relation The relation for which we need an array of names.
 * \param[in] names    The set of names for each element.
 * \return An array of strings with one string per constraint matrix column.
 */
static
char ** osl_relation_strings(osl_relation_p relation, osl_names_p names) {
  char ** strings;
  char temp[OSL_MAX_STRING];
  int i, offset;
  
  if ((relation == NULL) || (names == NULL)) {
    OSL_debug("no names or relation to build the name array");
    return NULL;
  }

  OSL_malloc(strings, char **, (relation->nb_columns + 1)*sizeof(char *));
  strings[relation->nb_columns] = NULL;

  // 1. Equality/inequality marker.
  OSL_strdup(strings[0], "e/i");
  offset = 1;

  // 2. Output dimensions.
  if (osl_relation_is_access(relation)) {
    // The first output dimension is the array name.
    OSL_strdup(strings[offset], "Arr");
    // The other ones are the array dimensions [1]...[n]
    for (i = offset + 1; i < relation->nb_output_dims + offset; i++) {
      sprintf(temp, "[%d]", i - 1);
      OSL_strdup(strings[i], temp);
    }
  }
  else
  if ((relation->type == OSL_TYPE_DOMAIN) ||
      (relation->type == OSL_TYPE_CONTEXT)) {
    for (i = offset; i < relation->nb_output_dims + offset; i++) {
      OSL_strdup(strings[i], names->iterators->string[i - offset]);
    }
  }
  else {
    for (i = offset; i < relation->nb_output_dims + offset; i++) {
      OSL_strdup(strings[i], names->scatt_dims->string[i - offset]);
    }
  }
  offset += relation->nb_output_dims;

  // 3. Input dimensions.
  for (i = offset; i < relation->nb_input_dims + offset; i++)
    OSL_strdup(strings[i], names->iterators->string[i - offset]);
  offset += relation->nb_input_dims;

  // 4. Local dimensions.
  for (i = offset; i < relation->nb_local_dims + offset; i++)
    OSL_strdup(strings[i], names->local_dims->string[i - offset]);
  offset += relation->nb_local_dims;

  // 5. Parameters.
  for (i = offset; i < relation->nb_parameters + offset; i++)
    OSL_strdup(strings[i], names->parameters->string[i - offset]);
  offset += relation->nb_parameters;

  // 6. Scalar.
  OSL_strdup(strings[offset], "1");

  return strings;
}



/**
 * osl_relation_column_string function:
 * this function returns an OpenScop comment string showing all column
 * names. It is designed to nicely fit a constraint matrix that would be
 * printed just below this line.
 * \param[in] relation The relation related to the comment line to build.
 * \param[in] strings  Array of textual names of the various elements.
 * \return A fancy comment string with all the dimension names.
 */
static
char * osl_relation_column_string(osl_relation_p relation, char ** strings) {
  int i, j;
  int index_output_dims;
  int index_input_dims;
  int index_local_dims;
  int index_parameters;
  int index_scalar;
  int space, length, left, right;
  char * scolumn;
  char temp[OSL_MAX_STRING];

  OSL_malloc(scolumn, char *, OSL_MAX_STRING);
 
  index_output_dims = 1;
  index_input_dims  = index_output_dims + relation->nb_output_dims;
  index_local_dims  = index_input_dims  + relation->nb_input_dims;
  index_parameters  = index_local_dims  + relation->nb_local_dims;
  index_scalar      = index_parameters  + relation->nb_parameters;

  // 1. The comment part.
  sprintf(scolumn, "#");
  for (j = 0; j < (OSL_FMT_LENGTH - 1)/2 - 1; j++)
    strcat(scolumn, " ");

  i = 0;
  while (strings[i] != NULL) {
    space  = OSL_FMT_LENGTH;
    length = (space > (int)strlen(strings[i])) ? (int)strlen(strings[i]) : space;
    right  = (space - length + (OSL_FMT_LENGTH % 2)) / 2;
    left   = space - length - right;

    // 2. Spaces before the name
    for (j = 0; j < left; j++)
      strcat(scolumn, " ");

    // 3. The (abbreviated) name
    for (j = 0; j < length - 1; j++) {
      sprintf(temp, "%c", strings[i][j]);
      strcat(scolumn, temp);
    }
    if (length >= (int)strlen(strings[i]))
      sprintf(temp, "%c", strings[i][j]);
    else 
      sprintf(temp, ".");
    strcat(scolumn, temp);

    // 4. Spaces after the name
    for (j = 0; j < right; j++)
      strcat(scolumn, " ");
    
    i++;
    if ((i == index_output_dims) ||
        (i == index_input_dims)  ||
        (i == index_local_dims)  ||
        (i == index_parameters)  ||
        (i == index_scalar))
      strcat(scolumn, "|");
    else
      strcat(scolumn, " ");
  }
  strcat(scolumn, "\n");

  return scolumn;
}


/**
 * osl_relation_is_simple_output function:
 * this function returns 1 or -1 if a given constraint row of a relation
 * corresponds to an output, 0 otherwise. We call a simple output an equality
 * constraint where exactly one output coefficient is not 0 and is either
 * 1 (in this case the function returns 1) or -1 (in this case the function
 * returns -1). 
 * \param[in] relation The relation to test for simple output.
 * \param[in] row      The row corresponding to the constraint to test.
 * \return 1 or -1 if the row is a simple output, 0 otherwise.
 */
static
int osl_relation_is_simple_output(osl_relation_p relation, int row) {
  int i;
  int first = 1;
  int sign  = 0;

  if ((relation == NULL) ||
      (relation->m == NULL) ||
      (relation->nb_output_dims == 0))
    return 0;

  if ((row < 0) || (row > relation->nb_rows))
    OSL_error("the specified row does not exist in the relation");

  // The constraint must be an equality.
  if (!osl_int_zero(relation->precision, relation->m[row][0]))
    return 0;

  // Check the output part has one and only one non-zero +1 or -1 coefficient.
  first = 1;
  for (i = 1; i <= relation->nb_output_dims; i++) {
    if (!osl_int_zero(relation->precision, relation->m[row][i])) {
      if (first)
        first = 0;
      else
        return 0;

      if (osl_int_one(relation->precision, relation->m[row][i]))
        sign = 1;
      else if (osl_int_mone(relation->precision, relation->m[row][i]))
        sign = -1;
      else
        return 0;
    }
  }

  return sign;
}


/**
 * osl_relation_expression_element function:
 * this function returns a string containing the printing of a value (e.g.,
 * an iterator with its coefficient or a constant).
 * \param[in]     val       Coefficient or constant value.
 * \param[in]     precision The precision of the value.
 * \param[in,out] first     Pointer to a boolean set to 1 if the current value
 *                          is the first of an expresion, 0 otherwise (maybe
 *                          updated).
 * \param[in]     cst       A boolean set to 1 if the value is a constant,
 *                          0 otherwise.
 * \param[in]     name      String containing the name of the element.
 * \return A string that contains the printing of a value.
 */
static
char * osl_relation_expression_element(osl_int_t val,
                                       int precision, int * first,
                                       int cst, char * name) {
  char * temp, * body, * sval;
 
  OSL_malloc(temp, char *, OSL_MAX_STRING * sizeof(char));
  OSL_malloc(body, char *, OSL_MAX_STRING * sizeof(char));
  OSL_malloc(sval, char *, OSL_MAX_STRING * sizeof(char));

  body[0] = '\0';
  sval[0] = '\0';

  // statements for the 'normal' processing.
  if (!osl_int_zero(precision, val) && (!cst)) {
    if ((*first) || osl_int_neg(precision, val)) {
      if (osl_int_one(precision, val)) {         // case 1
        sprintf(sval, "%s", name);
      }
      else {
        if (osl_int_mone(precision, val)) {      // case -1
          sprintf(sval, "-%s", name);
        }
	else {                                      // default case
	  osl_int_sprint_txt(sval, precision, val);
	  sprintf(temp, "*%s", name);
	  strcat(sval, temp);
        }
      }
      *first = 0;
    }
    else {
      if (osl_int_one(precision, val)) {
        sprintf(sval, "+%s", name);
      }
      else {
        sprintf(sval, "+");
	osl_int_sprint_txt(temp, precision, val);
	strcat(sval, temp);
	sprintf(temp, "*%s", name);
	strcat(sval, temp);
      }
    }
  }
  else {
    if (cst) {
      if ((osl_int_zero(precision, val) && (*first)) ||
          (osl_int_neg(precision, val)))
        osl_int_sprint_txt(sval, precision, val);
      if (osl_int_pos(precision, val)) {
        if (!(*first)) {
          sprintf(sval, "+");
          osl_int_sprint_txt(temp, precision, val);
	  strcat(sval, temp);
	}
	else {
          osl_int_sprint_txt(sval, precision, val);
        }
      }
    }
  }
  free(temp);
  free(body);

  return(sval);
}



/**
 * osl_relation_subexpression function:
 * this function returns a string corresponding to an affine (sub-)expression
 * stored at the "row"^th row of the relation pointed by "relation" between
 * the start and stop columns. Optionally it may oppose the whole expression.
 * \param[in] relation A set of linear expressions.
 * \param[in] row     The row corresponding to the expression.
 * \param[in] start   The first column for the expression (inclusive).
 * \param[in] stop    The last column for the expression (inclusive).
 * \param[in] oppose  Boolean set to 1 to negate the expression, 0 otherwise.
 * \param[in] strings Array of textual names of the various elements.
 * \return A string that contains the printing of an affine (sub-)expression.
 */
static
char * osl_relation_subexpression(osl_relation_p relation,
                                  int row, int start, int stop, int oppose,
                                  char ** strings) {
  int i, first = 1, constant;
  char * sval;
  char * sline;
  
  OSL_malloc(sline, char *, OSL_MAX_STRING * sizeof(char));
  sline[0] = '\0';

  // Create the expression. The constant is a special case.
  for (i = start; i <= stop; i++) {
    if (oppose) {
      osl_int_oppose(relation->precision,
                     &relation->m[row][i], relation->m[row][i]);
    }

    if (i == relation->nb_columns - 1)
      constant = 1;
    else
      constant = 0;

    sval = osl_relation_expression_element(relation->m[row][i],
        relation->precision, &first, constant, strings[i]);
    
    if (oppose) {
      osl_int_oppose(relation->precision,
                     &relation->m[row][i], relation->m[row][i]);
    }
    strcat(sline, sval);
    free(sval);
  }

  return sline;
}



/**
 * osl_relation_sprint_comment function:
 * this function prints into a string a comment corresponding to a constraint
 * of a relation, according to its type, then it returns this string. This
 * function does not check that printing the comment is possible (i.e., are
 * there enough names ?), hence it is the responsibility of the user to ensure
 * he/she can call this function safely.
 * \param[in] relation The relation for which a comment has to be printed.
 * \param[in] row      The constrain row for which a comment has to be printed.
 * \param[in] strings  Array of textual names of the various elements.
 * \param[in] arrays   Array of textual identifiers of the arrays.
 * \return A string which contains the comment for the row.
 */ 
static
char * osl_relation_sprint_comment(osl_relation_p relation, int row,
                                   char ** strings, char ** arrays) {
  int sign;
  size_t high_water_mark = OSL_MAX_STRING;
  char * string = NULL;
  char * expression;
  char buffer[OSL_MAX_STRING];

  OSL_malloc(string, char *, high_water_mark * sizeof(char));
  string[0] = '\0';
  
  if ((relation == NULL) || (strings == NULL)) {
    OSL_debug("no relation or names while asked to print a comment");
    return string;
  }
  
  if ((sign = osl_relation_is_simple_output(relation, row))) {
    // First case : output == expression.

    expression = osl_relation_subexpression(relation, row,
                                            1, relation->nb_output_dims,
                                            sign < 0,
                                            strings);
    snprintf(buffer, OSL_MAX_STRING, "   ## %s", expression);
    osl_util_safe_strcat(&string, buffer, &high_water_mark);
    free(expression);
    
    // We don't print the right hand side if it's an array identifier.
    if (!osl_relation_is_access(relation) ||
        osl_int_zero(relation->precision, relation->m[row][1])) {
      expression = osl_relation_subexpression(relation, row,
                                              relation->nb_output_dims + 1,
                                              relation->nb_columns - 1,
                                              sign > 0,
                                              strings);
      snprintf(buffer, OSL_MAX_STRING, " == %s", expression);
      osl_util_safe_strcat(&string, buffer, &high_water_mark);
      free(expression);
    }
    else {
      snprintf(buffer, OSL_MAX_STRING, " == %s",
               arrays[osl_relation_get_array_id(relation) - 1]);
      osl_util_safe_strcat(&string, buffer, &high_water_mark);
    }
  }
  else {
    // Second case : general case.
    
    expression = osl_relation_expression(relation, row, strings);
    snprintf(buffer, OSL_MAX_STRING, "   ## %s", expression);
    osl_util_safe_strcat(&string, buffer, &high_water_mark);
    free(expression);
    
    if (osl_int_zero(relation->precision, relation->m[row][0]))
      snprintf(buffer, OSL_MAX_STRING, " == 0");
    else
      snprintf(buffer, OSL_MAX_STRING, " >= 0");
    osl_util_safe_strcat(&string, buffer, &high_water_mark);
  }

  return string;
}


/**
 * osl_relation_column_string_scoplib function:
 * this function returns an OpenScop comment string showing all column
 * names. It is designed to nicely fit a constraint matrix that would be
 * printed just below this line.
 * \param[in] relation The relation related to the comment line to build.
 * \param[in] strings  Array of textual names of the various elements.
 * \return A fancy comment string with all the dimension names.
 */
static
char * osl_relation_column_string_scoplib(osl_relation_p relation,
                                          char ** strings) {
  int i, j;
  int index_output_dims;
  int index_input_dims;
  int index_local_dims;
  int index_parameters;
  int index_scalar;
  int space, length, left, right;
  char * scolumn;
  char temp[OSL_MAX_STRING];

  OSL_malloc(scolumn, char *, OSL_MAX_STRING);
 
  index_output_dims = 1;
  index_input_dims  = index_output_dims + relation->nb_output_dims;
  index_local_dims  = index_input_dims  + relation->nb_input_dims;
  index_parameters  = index_local_dims  + relation->nb_local_dims;
  index_scalar      = index_parameters  + relation->nb_parameters;

  // 1. The comment part.
  sprintf(scolumn, "#");
  for (j = 0; j < (OSL_FMT_LENGTH - 1)/2 - 1; j++)
    strcat(scolumn, " ");

  i = 0;
  while (strings[i] != NULL) {
    
    if (i == 0 || 
        (relation->type != OSL_TYPE_DOMAIN && i >= index_input_dims) ||
        (relation->type == OSL_TYPE_DOMAIN && i <= index_output_dims) ||
        i >= index_parameters) {
      space  = OSL_FMT_LENGTH;
      length = (space > (int)strlen(strings[i])) ? (int)strlen(strings[i]) : space;
      right  = (space - length + (OSL_FMT_LENGTH % 2)) / 2;
      left   = space - length - right;

      // 2. Spaces before the name
      for (j = 0; j < left; j++)
        strcat(scolumn, " ");

      // 3. The (abbreviated) name
      for (j = 0; j < length - 1; j++) {
        sprintf(temp, "%c", strings[i][j]);
        strcat(scolumn, temp);
      }
      if (length >= (int)strlen(strings[i]))
        sprintf(temp, "%c", strings[i][j]);
      else 
        sprintf(temp, ".");
      strcat(scolumn, temp);

      // 4. Spaces after the name
      for (j = 0; j < right; j++)
        strcat(scolumn, " ");
        
      if ((i == index_output_dims-1) ||
          (i == index_input_dims-1)  ||
          (i == index_local_dims-1)  ||
          (i == index_parameters-1)  ||
          (i == index_scalar-1))
        strcat(scolumn, "|");
      else
        strcat(scolumn, " ");
    }
    
    i++;
  }
  strcat(scolumn, "\n");

  return scolumn;
}


/**
 * osl_relation_names function:
 * this function generates as set of names for all the dimensions
 * involved in a given relation.
 * \param[in] relation The relation we have to generate names for.
 * \return A set of generated names for the input relation dimensions.
 */
static
osl_names_p osl_relation_names(osl_relation_p relation) {
  int nb_parameters = OSL_UNDEFINED;
  int nb_iterators  = OSL_UNDEFINED;
  int nb_scattdims  = OSL_UNDEFINED;
  int nb_localdims  = OSL_UNDEFINED;
  int array_id      = OSL_UNDEFINED;

  osl_relation_get_attributes(relation, &nb_parameters, &nb_iterators,
                              &nb_scattdims, &nb_localdims, &array_id);
  
  return osl_names_generate("P", nb_parameters,
                            "i", nb_iterators,
                            "c", nb_scattdims,
                            "l", nb_localdims,
                            "A", array_id);
}


/**
 * osl_relation_nb_components function:
 * this function returns the number of component in the union of relations
 * provided as parameter.
 * \param[in] relation The input union of relations.
 * \return The number of components in the input union of relations.
 */
int osl_relation_nb_components(osl_relation_p relation) {
  int nb_components = 0;
  
  while (relation != NULL) {
    nb_components++;
    relation = relation->next;
  }

  return nb_components;
}


static void print_macros(FILE *file)
{
    fprintf(file, "/* Useful macros. */\n") ;
    fprintf(file,
	"#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))\n");
    fprintf(file,
	"#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))\n");
    fprintf(file, "#define max(x,y)    ((x) > (y) ? (x) : (y))\n") ; 
    fprintf(file, "#define min(x,y)    ((x) < (y) ? (x) : (y))\n\n") ; 
    fprintf(file, "#ifdef TIME \n#define IF_TIME(foo) foo; \n"
                  "#else\n#define IF_TIME(foo)\n#endif\n\n");
}

static void print_declarations(FILE *file, int n, char **names, int indentation)
{
    int i;

    for (i = 0; i < indentation; i++)
      fprintf(file, " ");
    fprintf(file, "int %s", names[0]);
    for (i = 1; i < n; i++)
	fprintf(file, ", %s", names[i]);
    fprintf(file, ";\n");
}

static void print_scattering_declarations(FILE *file, CloogProgram *program,
        int indentation)
{
    int i, j, found = 0;
    int nb_scatnames = 0;
    CloogNames *names = program->names;

    // Copy pointer only to those scatering names that do not duplicate
    // iterator names.
    char **scatnames = (char **) malloc(sizeof(char *) * names->nb_scattering);
    for (i = 0; i < names->nb_scattering; ++i) {
      for (j = 0; j < names->nb_iterators; ++j) {
        found = 0;
        if (strcmp(names->scattering[i], names->iterators[j]) == 0) {
          found = 1;
        }
      }
      if (!found) {
        // Save a pointer (intentional!) to the names in the new array.
        scatnames[nb_scatnames++] = names->scattering[i];
      }
    }

    if (nb_scatnames) {
        for (i = 0; i < indentation; i++)
            fprintf(file, " ");
        fprintf(file, "/* Scattering iterators. */\n");
        print_declarations(file, nb_scatnames, scatnames, indentation);
    }
    free(scatnames);
}

static void print_iterator_declarations(FILE *file, CloogProgram *program,
	CloogOptions *options)
{
    CloogNames *names = program->names;

    print_scattering_declarations(file, program, 2);
    if (names->nb_iterators) {
	fprintf(file, "  /* Original iterators. */\n");
	print_declarations(file, names->nb_iterators, names->iterators, 2);
    }
}

static void print_callable_preamble(FILE *file, CloogProgram *program,
	CloogOptions *options)
{
    int j;
    CloogBlockList *blocklist;
    CloogBlock *block;
    CloogStatement *statement;

    fprintf(file, "extern void hash(int);\n\n");

    print_macros(file);

    for (blocklist = program->blocklist; blocklist; blocklist = blocklist->next) {
	block = blocklist->block;
	for (statement = block->statement; statement; statement = statement->next) {
	    fprintf(file, "#define S%d(", statement->number);
	    if (block->depth > 0) {
		fprintf(file, "%s", program->names->iterators[0]);
		for(j = 1; j < block->depth; j++)
		    fprintf(file, ",%s", program->names->iterators[j]);
	    }
	    fprintf(file,") { hash(%d);", statement->number);
	    for(j = 0; j < block->depth; j++)
		fprintf(file, " hash(%s);", program->names->iterators[j]);
	    fprintf(file, " }\n");
	}
    }
    fprintf(file, "\nvoid test("); 
    if (program->names->nb_parameters > 0) {
	fprintf(file, "int %s", program->names->parameters[0]);
	for(j = 1; j < program->names->nb_parameters; j++)
	    fprintf(file, ", int %s", program->names->parameters[j]);
    }
    fprintf(file, ")\n{\n"); 
    print_iterator_declarations(file, program, options);
}

static void print_callable_postamble(FILE *file, CloogProgram *program)
{
    fprintf(file, "}\n"); 
}


static void print_comment(FILE *file, CloogOptions *options,
			  const char *fmt, ...)
{
  va_list args;

  va_start(args, fmt);
  if (options->language == CLOOG_LANGUAGE_FORTRAN) {
    fprintf(file, "! ");
    vfprintf(file, fmt, args);
    fprintf(file, "\n");
  } else {
    fprintf(file, "/* ");
    vfprintf(file, fmt, args);
    fprintf(file, " */\n");
  }
}

static LogicalResult emitOpenScop(ModuleOp module, llvm::raw_ostream &os) {
  llvm::SmallVector<std::unique_ptr<OslScop>, 8> scops;

  if (failed(translateModuleToOpenScop(module, scops, os)))
    return failure();  

  CloogProgram * program ;
  CloogOptions * options ;
  CloogState *state;
  CloogInput * input; 
  std::vector<std::string> readsAll;
  std::vector<std::string> writesAll;
  
  for (auto &scop : scops) {
    
    state = cloog_state_malloc();
    options = cloog_options_malloc(state);
    options->openscop = 1;
    osl_scop_print(stdout, scop->get());
    options->scop = scop->get();
    input = cloog_input_from_osl_scop(options->state, scop->get());
    cloog_options_copy_from_osl_scop(scop->get(), options);
    /* Reading the program informations. */
    program = cloog_program_alloc(input->context, input->ud, options); 
    struct osl_statement* statement;
    struct osl_relation_list* relation;
    struct osl_access* access;
    struct clast_stmt *root;
    root = cloog_clast_create(program, options);
    int indentation = 0;
    osl_names_p names = osl_scop_names(scop->get());
    
    // Traverse statements in the OpenScop representation
    for (statement = scop->get()->statement; statement!=NULL; statement = statement->next) {

        struct osl_relation_list* relation = statement->access;

        char* stmt_text;
        // osl_relation_list_pprint_elts(stdout, statement->access, names);
        size_t i;
        osl_relation_list_p head = relation;

        // Count the number of elements in the list with non-NULL content.
        i = osl_relation_list_count(relation);
        
        // Print each element of the relation list.
        if (i > 0) {

            i = 0;
            char* statementText;
            OSL_malloc(statementText, char *, 2048 * sizeof(char));
            statementText[0] = '\0';
            std::vector<std::string> reads;
            std::vector<std::string> writes;
            while (head) {
            char* statementAccess;
            OSL_malloc(statementAccess, char *, 2048 * sizeof(char));
            statementAccess[0] = '\0';
            char * type;
            

            if (head->elt != NULL) {
                // osl_relation_pprint(stdout, head->elt, names);
                // char * string = osl_relation_spprint(head->elt, names);
                size_t high_water_mark = 2048;
                char * string = NULL;
                char * temp;
                char buffer[OSL_MAX_STRING];
                OSL_malloc(string, char *, high_water_mark * sizeof(char));
                string[0] = '\0';

                if (osl_relation_nb_components(head->elt) > 0) {
                    type = osl_relation_sprint_type(head->elt);
                    
                    osl_util_safe_strcat(&string, type, &high_water_mark);
                    
                    
                    snprintf(buffer, OSL_MAX_STRING, "\n");
                    osl_util_safe_strcat(&string, buffer, &high_water_mark);

                    // temp = osl_relation_spprint_polylib(head->elt, names);
                    int i_, j;
                    int part, nb_parts;
                    int generated_names = 0;
                    high_water_mark = OSL_MAX_STRING;
                    char * string2 = NULL;
                    char buffer3[OSL_MAX_STRING];
                    char ** name_array = NULL;
                    char * scolumn;
                    char * comment;
                    // osl_relation_p relation = head->elt;

                    // if (relation == NULL)
                    //     return osl_util_strdup("# NULL relation\n");

                    OSL_malloc(string2, char *, high_water_mark * sizeof(char));
                    string2[0] = '\0';

                    // Generates the names for the comments if necessary.
                    // if (names == NULL) {
                    //     generated_names = 1;
                    //     names = osl_relation_names(head->elt);
                    // }

                    nb_parts = osl_relation_nb_components(head->elt);

                    if (nb_parts > 1) {
                        snprintf(buffer3, OSL_MAX_STRING, "# Union with %d parts\n%d\n",
                                nb_parts, nb_parts);
                        osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                    }

                    // Print each part of the union.
                    for (part = 1; part <= nb_parts; part++) {
                        // Prepare the array of strings for comments.
                        name_array = osl_relation_strings(head->elt, names);

                        if (nb_parts > 1) {
                        snprintf(buffer3, OSL_MAX_STRING, "# Union part No.%d\n", part);
                        osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                        }

                        snprintf(buffer3, OSL_MAX_STRING, "%d %d %d %d %d %d\n",
                                head->elt->nb_rows,        head->elt->nb_columns,
                                head->elt->nb_output_dims, head->elt->nb_input_dims,
                                head->elt->nb_local_dims,  head->elt->nb_parameters);
                        osl_util_safe_strcat(&string2, buffer3, &high_water_mark);

                        if (head->elt->nb_rows > 0) {
                        scolumn = osl_relation_column_string(head->elt, name_array);
                        snprintf(buffer3, OSL_MAX_STRING, "%s", scolumn);
                        osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                        free(scolumn);
                        }

                        for (i_ = 0; i_ < head->elt->nb_rows; i_++) {
                        for (j = 0; j < head->elt->nb_columns; j++) {
                            osl_int_sprint(buffer3, head->elt->precision, head->elt->m[i_][j]);
                            osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                            snprintf(buffer, OSL_MAX_STRING, " ");
                            osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                        }
                        char * string3 = NULL;
                        if (name_array != NULL) {
                            // comment = osl_relation_sprint_comment(head->elt, i_, name_array,
                            //                                     names->arrays->string);
                            char buffer2[OSL_MAX_STRING];  
                            char** arrays = names->arrays->string;   
                            char * expression; 

                            OSL_malloc(string3, char *, high_water_mark * sizeof(char));
                            
                            string3[0] = '\0'; 
                            int sign;
                            if ((sign = osl_relation_is_simple_output(head->elt, i_))) {
                                // First case : output == expression.

                                expression = osl_relation_subexpression(head->elt, i_,
                                                                        1, head->elt->nb_output_dims,
                                                                        sign < 0,
                                                                        name_array);
                                snprintf(buffer2, OSL_MAX_STRING, "   ## %s", expression);
                                // osl_util_safe_strcat(&string3, buffer2, &high_water_mark);
                                free(expression);
                                
                                // We don't print the right hand side if it's an array identifier.
                                if (!osl_relation_is_access(head->elt) ||
                                    osl_int_zero(head->elt->precision, head->elt->m[i_][1])) {
                                expression = osl_relation_subexpression(head->elt, i_,
                                                                        head->elt->nb_output_dims + 1,
                                                                        head->elt->nb_columns - 1,
                                                                        sign > 0,
                                                                        name_array);
                                snprintf(buffer2, OSL_MAX_STRING, "[%s]", expression);
                                osl_util_safe_strcat(&string3, buffer2, &high_water_mark);
                                free(expression);
                                }
                                else {
                                snprintf(buffer2, OSL_MAX_STRING, "%s",
                                        arrays[osl_relation_get_array_id(head->elt) - 1]);
                                osl_util_safe_strcat(&string3, buffer2, &high_water_mark);
                                }
                            }
                            else {
                                // Second case : general case.
                                
                                expression = osl_relation_expression(head->elt, i_,  name_array);
                                snprintf(buffer2, OSL_MAX_STRING, "   ## %s", expression);
                                osl_util_safe_strcat(&string3, buffer2, &high_water_mark);
                                free(expression);
                                
                                if (osl_int_zero(head->elt->precision, head->elt->m[i_][0]))
                                snprintf(buffer2, OSL_MAX_STRING, " == 0");
                                else
                                snprintf(buffer2, OSL_MAX_STRING, " >= 0");
                                osl_util_safe_strcat(&string3, buffer2, &high_water_mark);
                            }

                            
                        }
                        printf("Comment: %s\n", string3);
                        if(string3 != NULL) {   
                            strcat(statementAccess, string3); 
                            printf("Statement Access: %s\n", statementAccess);
                        }
                            
                        snprintf(buffer3, OSL_MAX_STRING, "\n");
                        osl_util_safe_strcat(&string2, buffer3, &high_water_mark);
                        }

                        // Free the array of strings.
                        if (name_array != NULL) {
                            for (i_ = 0; i_ < head->elt->nb_columns; i_++)
                                free(name_array[i_]);
                            free(name_array);
                        }

                        head->elt = head->elt->next;
                    }
                    
                    if (generated_names)
                        osl_names_free(names);
                    temp = string2;
                    osl_util_safe_strcat(&string, temp, &high_water_mark);
                    free(temp);
                }
                fprintf(stdout, "%s", string);
                free(string);
                if (head->next != NULL)
                fprintf(stdout, "\n");
                i++;
            }
            head = head->next;
            char* reversedAccess;
            OSL_malloc(reversedAccess, char *, 2048 * sizeof(char));
            reversedAccess[0] = '\0';
            rearrangeString(statementAccess, reversedAccess);
            
            if(strcmp(type, "READ") == 0) {
                  printf("\nRead Access: ");
                  reads.push_back(reversedAccess);
                  readsAll.push_back(reversedAccess);
                  fprintf(stdout, "%s", reversedAccess);

            } else if(strcmp(type, "WRITE") == 0){
                  printf("\nWrite Access: ");
                  writes.push_back(reversedAccess);
                  writesAll.push_back(reversedAccess);
                  fprintf(stdout, "%s", reversedAccess);
            }
            free(type);
            strcat(statementText, reversedAccess);
            // fprintf(stdout, "Statement Text: %s", statementText);
            free(statementAccess);
            free(reversedAccess);
            }

            fprintf(stdout, "STMT: %s", statementText);

            int add_fakeiter = statement->domain->nb_rows == 0 &&
                   statement->scattering->nb_rows == 1;
            osl_body_p body = (osl_body_p)osl_generic_lookup(statement->extension, OSL_URI_BODY);

            std::string str;
            char *exp;
            char *cstr;

            if(reads.size() > 0 && writes.size() > 0) {
                str = generateStatement(reads, writes, std::string(statementText));
               //std::cout << "GENERATED STATEMENT: " << str << std::endl;
                cstr = new char[str.length() + 1];
                strcpy(cstr, str.c_str());
                exp = (char*)cstr;
            } else {
                strcat(statementText, ";");
                exp =  (char*)statementText;
            }
                
            
            // Read the body: 
            printf("Body: %s\n", exp);

            // Insert the body.
            body->expression = osl_strings_encapsulate(exp);
            osl_body_print_scoplib(stdout, body); // Print the body

            statement->extension->data = body;
            osl_arrays_p arrays = (osl_arrays_p)osl_generic_lookup(scop->get()->extension, OSL_URI_ARRAYS);

            osl_arrays_dump(stdout, arrays);
            body = NULL; 
        }
        else {
            fprintf(stdout, "# NULL relation list\n");
        }
        
        // osl_relation_print(stdout, relation->elt); // Print the access relation 
    }

    // print_scop_to_c(stdout, scop->get());
    break;
  } 
 
  /* Generating and printing the code. */
  FILE *fp = NULL;
  fp = fopen("./scop.c" ,"w+");
  options->compilable = 1; 
  program->language = 'c';
  options->callable = 1;
  program = cloog_program_generate(program,options);
  
  if (options->structure)
    cloog_program_print(fp,program) ;
//   cloog_program_pprint(stdout,program,options) ;
    int i, j, indentation = 0;
    CloogStatement * statement ;
    CloogBlockList * blocklist ;
    CloogBlock * block ;
    struct clast_stmt *root;

    // if (cloog_program_osl_pprint(stdout, program, options))
    //     return success();

    if (program->language == 'f')
        options->language = CLOOG_LANGUAGE_FORTRAN ;
    else
        options->language = CLOOG_LANGUAGE_C ;
    
    #ifdef CLOOG_RUSAGE
    print_comment(fp, options, "Generated from %s by %s in %.2fs.",
            options->name, cloog_version(), options->time);
    #else
    print_comment(fp, options, "Generated from %s by %s.",
            options->name, cloog_version());
    #endif
    #ifdef CLOOG_MEMORY
    print_comment(fp, options, "CLooG asked for %d KBytes.", options->memory);
    cloog_msg(CLOOG_INFO, "%.2fs and %dKB used for code generation.\n",
        options->time,options->memory);
    #endif
    
    /* If the option "compilable" is set, we provide the whole stuff to generate
    * a compilable code. This code just do nothing, but now the user can edit
    * the source and set the statement macros and parameters values.
    */
    if (options->compilable && (program->language == 'c'))
    { /* The headers. */
        fprintf(fp,"/* DON'T FORGET TO USE -lm OPTION TO COMPILE. */\n\n") ;
        fprintf(fp,"/* Useful headers. */\n") ;
        fprintf(fp,"#include <stdio.h>\n") ;
        fprintf(fp,"#include <stdlib.h>\n") ;
        fprintf(fp,"#include <math.h>\n\n") ;

        /* The value of parameters. */
        fprintf(fp,"/* Parameter value. */\n") ;
        for (i = 1; i <= program->names->nb_parameters; i++)
        fprintf(fp, "#define PARVAL%d %d\n", i, options->compilable);
        
        /* The macros. */
        print_macros(fp);

        /* The statement macros. */
        fprintf(fp,"/* Statement macros (please set). */\n") ;
        blocklist = program->blocklist ;
        while (blocklist != NULL)
        { block = blocklist->block ;
        statement = block->statement ;
        while (statement != NULL)
        { fprintf(fp,"#define S%d(",statement->number) ;
            if (block->depth > 0)
            { fprintf(fp,"%s",program->names->iterators[0]) ;
            for(j=1;j<block->depth;j++)
            fprintf(fp,",%s",program->names->iterators[j]) ;
            }
            fprintf(fp,") {total++;") ;
        if (block->depth > 0) {
            fprintf(fp, " printf(\"S%d %%d", statement->number);
            for(j=1;j<block->depth;j++)
                fprintf(fp, " %%d");
            
            fprintf(fp,"\\n\",%s",program->names->iterators[0]) ;
        for(j=1;j<block->depth;j++)
            fprintf(fp,",%s",program->names->iterators[j]) ;
            fprintf(fp,");") ;
            }
            fprintf(fp,"}\n") ;
            
        statement = statement->next ;
        }
        blocklist = blocklist->next ;
        }
        
        /* The iterator and parameter declaration. */
        fprintf(fp,"\nint main() {\n") ; 
        print_iterator_declarations(fp, program, options);
        if (program->names->nb_parameters > 0)
        { fprintf(fp,"  /* Parameters. */\n") ;
        fprintf(fp, "  int %s=PARVAL1",program->names->parameters[0]);
        for(i=2;i<=program->names->nb_parameters;i++)
            fprintf(fp, ", %s=PARVAL%d", program->names->parameters[i-1], i);
        
        fprintf(fp,";\n");
        }
        fprintf(fp,"  int total=0;\n");
        generateDeclarations(readsAll, writesAll);
        int maxBound = 2000;
        for(auto it = arraySizes.begin(); it != arraySizes.end(); ++it) {
            fprintf(fp, "  int %s", it->first.c_str());
            for(int p=0; p < it->second; p++) {
                fprintf(fp, "[%d]",  maxBound);
            }
            fprintf(fp, ";\n");
        }
        fprintf(fp,"\n") ; 
        /* And we adapt the identation. */
        indentation += 2 ;
    } else if (options->callable && program->language == 'c') {
        print_callable_preamble(fp, program, options);
        indentation += 2;
    }
    fprintf(fp, "#pragma scop\n");
    root = cloog_clast_create(program, options);
    clast_pprint(fp, root, indentation, options);
    cloog_clast_free(root);
    fprintf(fp, "#pragma endscop\n");

    /* The end of the compilable code in case of 'compilable' option. */
    if (options->compilable && (program->language == 'c'))
    {
        fprintf(fp, "\n  printf(\"Number of integral points: %%d.\\n\",total);");
        fprintf(fp, "\n  return 0;\n}\n");
    } else if (options->callable && program->language == 'c')
        print_callable_postamble(fp, program); 
  cloog_program_free(program) ;   
  fclose(fp);
  return success();
}

// The main function that implements the BullsEye based analysis  
static mlir::func::FuncOp bullseyeCacheMisses(mlir::func::FuncOp f) {

    LLVM_DEBUG(dbgs() << "BullsEye Analysis: \n");
    LLVM_DEBUG(f.dump());

    std::vector<char *> Arguments;
    char Argument1[] = "program";
    char ArgumentI[] = "-I";
    Arguments.push_back(Argument1); 
    int ArgumentCount = Arguments.size();
    struct pet_options *options_;
    options_ = pet_options_new_with_defaults();
    ArgumentCount = pet_options_parse(options_, ArgumentCount, &Arguments[0], ISL_ARG_ALL);
    isl_ctx* ctx = isl_ctx_alloc_with_options(&pet_options_args, options_);
    
    // default
    const int CACHE_LINE_SIZE = 64;
    const int CACHE_SIZE3 = 1024 * 1024;
    const int CACHE_SIZE2 = 512 * 1024;
    const int CACHE_SIZE1 = 32 * 1024;

    // Call bullseye  
    std::vector<std::string> IncludePaths;
    std::vector<std::string> defineparameters;
    isl::ctx Context = allocateContextWithIncludePaths(IncludePaths); 
    std::vector<long> CacheSizes = {CACHE_SIZE1, CACHE_SIZE2, CACHE_SIZE3};
    bullseyelib::ProgramParameters Parameters("./scop.c",
        CacheSizes, CACHE_LINE_SIZE, IncludePaths, defineparameters, "", false); 

    bullseyelib::CacheMissResults cmc = bullseyelib::run_model_new(Context, Parameters); 
    std::vector<long> cap_misses = cmc.TotalCapacity;

    // Add the attributes to the function
    mlir::FloatType floatType = mlir::FloatType::getF32(f->getContext());
    mlir::IntegerType intType = mlir::IntegerType::get(f->getContext(), 64); 
    mlir::IntegerAttr* intAttr = new mlir::IntegerAttr[cap_misses.size()];

    int attr_c = 0;
    std::string level = "";
    for(double cp : cap_misses) {
        std::cout << "Capacity Misses: " << cp << "\n"; 
        intAttr[attr_c] = mlir::IntegerAttr::get(intType, cp);
        level = "L" + std::to_string(attr_c+1);
        f->setAttr(level, intAttr[attr_c++]);
    }
    
    // Create the float attribute
    mlir::FloatAttr floatAttr1 = mlir::FloatAttr::get(floatType, cmc.TotalCompulsory); 
    // Set value of the attribute
    f->setAttr("compulsory", floatAttr1);

    mlir::FloatAttr floatAttr2 = mlir::FloatAttr::get(floatType, cmc.TotalAccesses);
    f->setAttr("accesses", floatAttr2);  
 
    return f;
}


std::pair<unsigned, unsigned> countMemoryAccess(Operation *op, unsigned &memoryAccess, std::vector<mlir::Value> &funcArgs, unsigned &arithOp, 
                          std::unordered_map<mlir::Value, int64_t> &inductVarDict) {
    // Helper lambda to calculate total trip count from induction variable dictionary
    auto getTripCount = [&inductVarDict]() -> int64_t {
        int64_t tripCount = 1;
        for (auto const& [key, val] : inductVarDict) {
            tripCount *= val;
        }
        return tripCount;
    };

    // Process the operation
    if (isa<mlir::affine::AffineLoadOp>(op) || isa<mlir::affine::AffineStoreOp>(op)) {
        int64_t tripCount = getTripCount();
        memoryAccess += tripCount; // Simplified logic, adjust for off-chip/on-chip memory
    } else if (isa<mlir::arith::AddIOp>(op) || isa<mlir::arith::SubIOp>(op) ||
                isa<mlir::arith::MulIOp>(op) || isa<mlir::arith::AddFOp>(op) ||
                isa<mlir::arith::SubFOp>(op) || isa<mlir::arith::MulFOp>(op)) {
        int64_t tripCount = getTripCount();
        arithOp += tripCount; // Increase for each arithmetic operation inside loops
    } else if (auto forOp = dyn_cast<mlir::affine::AffineForOp>(op)) {
        // Get loop bounds and step for the AffineForOp 
        int64_t step = forOp.getStep();

        // Get the lower bound value
        auto lbResult = forOp.getLowerBoundMap().getSingleConstantResult();

        // Get the upper bound value
        auto ubResult = forOp.getUpperBoundMap().getSingleConstantResult();

        // Calculate the trip count
        int64_t tripCount = abs(ubResult - lbResult) / step;
        llvm::outs() << "Trip Count: " << tripCount << "\n";
        
        // Save old trip count for the induction variable and update it
        mlir::Value indVar = forOp.getInductionVar();
        int64_t oldTripCount;
        if (inductVarDict.find(indVar) != inductVarDict.end()) {
            llvm::outs() << "Found\n";
            oldTripCount = inductVarDict[indVar];
        } else {
            llvm::outs() << "Not Found\n";
            oldTripCount = 1;
        }
        // int64_t oldTripCount = std::count(inductVarDict.begin(), inductVarDict.end(), Compare(indVar)) ? inductVarDict[indVar] : 1;
        inductVarDict[indVar] = tripCount;

        // Recursively process inner operations
        forOp.getRegion().walk([&](mlir::Operation* op) {
            countMemoryAccess(op, memoryAccess, funcArgs, arithOp, inductVarDict);
        });
        

        // Restore old trip count for the induction variable
        inductVarDict[indVar] = oldTripCount;
    } else if (auto ifOp = dyn_cast<mlir::affine::AffineIfOp>(op)) {
        // Example: processing AffineIfOp would go here, similar to AffineForOp
        // This would involve interpreting conditions and potentially modifying trip counts,
        // which requires parsing the condition, a non-trivial task and hence omitted here.
        int64_t reductionFactor = 2; // Simplified, as your actual logic may vary

        // Apply reduction to all relevant trip counts
        std::unordered_map<mlir::Value, int64_t> reducedInductVarDict = inductVarDict;
        for (auto &pair : reducedInductVarDict) {
            pair.second -= reductionFactor; // Reduce trip count by some factor
            pair.second = std::max(pair.second, int64_t(1)); // Ensure trip count doesn't go below 1
        }

        // Recursively process operations inside the 'then' and 'else' blocks, if any
        if (ifOp.getThenBlock()) {
            for (Operation &nestedOp : ifOp.getThenBlock()->getOperations()) {
                countMemoryAccess(&nestedOp, memoryAccess, funcArgs, arithOp, reducedInductVarDict);
            }
        }
        if (ifOp.getElseBlock()) {
            for (Operation &nestedOp : ifOp.getElseBlock()->getOperations()) {
                countMemoryAccess(&nestedOp, memoryAccess, funcArgs, arithOp, reducedInductVarDict);
            }
        }
    }
    // Extend with other operation types as necessary

    return std::make_pair(arithOp, memoryAccess);
}

void printAI(mlir::func::FuncOp &func) {
        unsigned numMemoryAccesses = 0;
        unsigned numArithOps = 0;
        std::vector<mlir::Value> funcArgs;

        // get function entry block arguments
        auto entryBlock = &func.getBlocks().front();
        auto entry_arguments = entryBlock->getArguments();

        // Add function arguments to the vector
        for (auto arg : entry_arguments) {
            funcArgs.push_back(arg);
        }        
        
        // Assuming the function's output is the last operand of its ReturnOp, if it exists
        mlir::Value outputArg;
        for (auto &block : func.getBlocks()) {
            for (auto &op : block) {
                if (isa<func::ReturnOp>(op)) {
                    if (op.getNumOperands() > 0) {
                        outputArg = op.getOperand(0);
                        funcArgs.push_back(outputArg);
                    }
                }
            }
        }

        // print funcArgs
        llvm::outs() << "Function Arguments: ";
        for (auto arg : funcArgs) {
            llvm::outs() << arg << " ";
        }
        llvm::outs() << "\n";

        llvm::outs() << "Function " << func.getName() << ":\n";

        std::unordered_map<mlir::Value, int64_t> inductVarDict; // For tracking loop trip counts

        // Walk through the operations in the function to count memory accesses and arithmetic operations
        for (auto &op : entryBlock->getOperations()) { 
            op.dump();
            std::pair<unsigned, unsigned> counter = countMemoryAccess(&op, numMemoryAccesses, funcArgs, numArithOps, inductVarDict);
            numArithOps = counter.first;
            numMemoryAccesses = counter.second;
        }

        llvm::outs() << "Memory Accesses: " << numMemoryAccesses*4 << "\n";
        llvm::outs() << "Arithmetic Operations: " << numArithOps << "\n\n";
        llvm::outs() << "Arithmetic Intensity: " << (double)numArithOps / (numMemoryAccesses*4) << "\n";

        // Create float type
        mlir::FloatType floatType = mlir::FloatType::getF32(func->getContext());
        // Create the float attribute
        mlir::FloatAttr floatAttr = mlir::FloatAttr::get(floatType, numArithOps); 
        // Set value of the attribute
        func->setAttr("flops", floatAttr);

        mlir::FloatAttr floatAttr2 = mlir::FloatAttr::get(floatType, numMemoryAccesses*4);
        func->setAttr("bytes", floatAttr2);
}


namespace {
    class BullsEyeAnalysisPass :
        public mlir::PassWrapper<BullsEyeAnalysisPass, 
                                 OperationPass<mlir::ModuleOp>> {

        public:
        BullsEyeAnalysisPass() = default;
        BullsEyeAnalysisPass(const BullsEyeAnalysisPass &pass) {}

        void runOnOperation() override {

                mlir::ModuleOp m = getOperation();

                mlir::OpBuilder b(m.getContext());
                llvm::raw_ostream *out = &llvm::errs();
                std::error_code EC;
                out = new raw_fd_ostream("", EC);
                emitOpenScop(m, *out); 
                if (out != &outs())
                     delete out;

                m.walk([&](mlir::func::FuncOp f) {
                    if (!f->getAttr("scop.stmt") && !f->hasAttr("scop.ignored")) {
                      f.dump();
                      f = bullseyeCacheMisses(f);
                    }
                    printAI(f);  
                });
 
        }
    };

}


void polymer::registerBullsEyeAnalysisPass() {
  PassPipelineRegistration<>(
      "cache-analysis", "Cache miss analysis for multi-level caches.",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<BullsEyeAnalysisPass>());
        pm.addPass(createCanonicalizerPass());
      });
}
