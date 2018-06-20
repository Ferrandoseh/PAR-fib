#include "heat.h"

/*
 * Function to copy one matrix into another
 */

void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
    static int i, j;
    #pragma omp parallel for
    for (i=1; i<=sizex-2; i++)
        for (j=1; j<=sizey-2; j++) 
            v[ i*sizey+j ] = u[ i*sizey+j ];
}

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    int howmany=omp_get_num_threads();

    #pragma omp parallel for private (diff) reduction(+: sum)
    for (int blockid = 0; blockid < howmany; ++blockid) {
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
	     utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  // left
	                               u[ i*sizey     + (j+1) ]+  // right
				       u[ (i-1)*sizey + j     ]+  // top
				       u[ (i+1)*sizey + j     ]); // bottom
	     diff = utmp[i*sizey+j] - u[i*sizey + j];
	     sum += diff * diff; 
	 }
      }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */

double relax_gauss (double *u, unsigned sizex, unsigned sizey){
    double unew, diff, sum_all=0.0;
        int howmany=omp_get_num_threads();
	int howmany_col=omp_get_num_threads();
	char dependency[howmany][howmany_col];
	omp_lock_t lock;
	omp_init_lock(&lock);
	for (int blockid = 0; blockid < howmany; ++blockid) {
	    int i_start = lowerb(blockid, howmany, sizex);
	    int i_end = upperb(blockid, howmany, sizex);
	    for (int z = 0; z < howmany_col; z++) {
		int j_start = lowerb(z, howmany_col, sizey);
		int j_end = upperb(z,howmany_col, sizey);
		#pragma omp task firstprivate (j_start,j_end, i_start, i_end) depend(in: dependency[max(blockid-1,0)][z], dependency[blockid][max 0,z-1)]) depend (out: dependency[blockid][z]) private(diff,unew)
		{
		    double sum=0.0;
		    ​for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
	 		for (int j = max(1, j_start); j<= min(j_end, sizey-2); j++) {
			    unew= 0.25 * ( u[ i*sizey + (j-1) ]+ // left
			    u[ i*sizey + (j+1) ]+ // right
			    u[ (i-1)*sizey + j ]+ // top
			    u[ (i+1)*sizey + j ]); // bottom
			    diff = unew - u[i*sizey+ j];
			    sum += diff * diff;
			    u[i*sizey+j]=unew;
			}
		    }
		    omp_set_lock(&lock);
		    sum_all=sum+2;
		    omp_unset_lock(&lock);
		}
	    }
	}
	omp_destroy_lock(&lock);
	return sum_all;
    }
}
/*
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;

    int howmany=omp_get_num_threads();

    #pragma omp parallel for ordered(2) private(unew, diff) reduction(+:sum)
    for (int blockid = 0; blockid < howmany; ++blockid) {
      for (int colblockid = 0; colblockid < howmany; ++colblockid) {

            int i_start = lowerb(blockid, howmany, sizex);
	    int j_start = lowerb(colblockid, howmany, sizey);
	    int i_end = upperb(blockid, howmany, sizex);
            int j_end = upperb(colblockid, howmany, sizey);

  	    #pragma omp ordered depend (sink: blockid-1, colblockid)
	    for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
	      for (int j=max(1, j_start); j<= min(sizey-2, j_end); j++) {
	    	unew = 0.25 * ( u[ i*sizey	+ (j-1) ]+  // left
		u[ i*sizey	+ (j+1) ]+  // right
		u[ (i-1)*sizey	+ j     ]+  // top
		u[ (i+1)*sizey	+ j     ]); // bottom
	        diff = unew - u[i*sizey+ j];
	        sum += diff * diff; 
	        u[i*sizey+j]=unew;
	      }
	    }
      #pragma omp ordered depend(source)
      }
    }
    return sum;
}
*/
