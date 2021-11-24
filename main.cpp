#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <queue>
#include <array>
#include <random>
#include <iostream>
#include <time.h>
#include <chrono>
#include <assert.h>
#include <algorithm>
#include <stack>

using namespace std;

int N;
int M;
int* neighbour; 
int* offset;
bool flag = true;

// ------------------ UTILS ------------------

void printGraph(int *n, int *o) {
	cout << "Printing graph\n";
	for (int i = 0; i < M; ++i)   cout << n[i] << " ";
	cout << "\n";
	for (int i = 0; i < N+1; ++i) cout << o[i]    << " ";
	cout << endl;
}

void ER_Generator(double p) {
	srand (time(NULL));
	vector<int> neighbour;
	bool connect;
	int offset[N+1];
	int shift = 0;
	for (int i = 0; i < N; i++) {
		offset[i] = shift;
		for (int j = i+1; j < N; ++j) {
			connect = (rand() % 100) + 1 < (p*100);
			if (connect) {
				neighbour.push_back(j+1);
				shift++;
			}
		}
		//connect with the previous ones
		for (int k = 0; k < i; k++)
			for (int m = offset[k]; m < offset[k+1]; ++m)
				if (neighbour[m] == i+1) {
					neighbour.push_back(k+1);
					shift++;
				}	
	}
	offset[N] = neighbour.size();
}

//Only works for undirected graphs
bool IsAdjacent(int u, int v) {
	int degree_u = offset[u+1] - offset[u];
	int degree_v = offset[v+1] - offset[v];
	//iterate over the vertex with less neighbours to spare time
	if (degree_u > degree_v) swap(u,v);
	for (int i = offset[u]; i < offset[u+1]; i++) if (neighbour[i] == v) return true;
	return false;
}

//For directed graphs (or undirected but less efficient)
bool IsAdjacentDir(int u, int v) { // u--->v
	for (int i = offset[u]; i < offset[u+1]; i++) if (neighbour[i] == v) return true;
	return false;
}

bool mysort(const pair<int,int> &a, const pair<int,int> &b) { 
    if (a.first < b.first) return true;
    else if (a.first == b.first) return a.second < b.second;
    else return false;
} 

void link_selection_model() {
	vector<pair<int,int> > edgelist;
	edgelist.push_back(pair<int,int>(0, 1));
	edgelist.push_back(pair<int,int>(1, 0));
	edgelist.push_back(pair<int,int>(1, 2));
	edgelist.push_back(pair<int,int>(2, 1));
	edgelist.push_back(pair<int,int>(2, 0));
	edgelist.push_back(pair<int,int>(0, 2));
	int size = edgelist.size();
	pair<int,int> random_edge;
	srand(time(NULL));

	// number of nodes
	int nodes = 103;

	for (int i=3; i<nodes; i++) {
		random_edge = edgelist[rand() % size++];
		edgelist.push_back(pair<int,int>(i, random_edge.first));
		edgelist.push_back(pair<int,int>(random_edge.first, i));
		edgelist.push_back(pair<int,int>(random_edge.second, i));
		edgelist.push_back(pair<int,int>(i, random_edge.second));
	}

	sort(edgelist.begin(), edgelist.end(), mysort);

	int* neighbour2 = (int*) malloc(sizeof(int)*edgelist.size()); 
	int* offset2    = (int*) malloc(sizeof(int)*(nodes+1)); // +1 is a hack to know that we are at the end of the array

	int u = 0; int v = 0;
	int previous = 0;
	int shift = 0;
	offset2[0] = 0;
	for (int i = 0; i < edgelist.size(); i++) {
		u = edgelist[i].first;
		v = edgelist[i].second;
		if (u == previous) {
			neighbour2[i] = v;
			shift++;
		}
		else {
			neighbour2[i] = v;
			for (int j = previous; j < u; j++)
				offset2[j+1] = shift;
			previous = u;
			shift++;
		}
	}
	offset2[nodes] = edgelist.size();
	
	return;
}


void printVec(vector< vector<unsigned long int> >& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << "Vertex = " << i << " ";
    	for (int j = 0; j < vec[i].size(); j++) {
        	cout << vec[i][j];
    	}
    	cout << "\n";
	}
	cout << "\n";
}

void printNeighbourhoods(int max_distance, vector<unsigned long int>& N_Function){
	cout << "\n";
	for (int i = 0; i < max_distance; i++) 
		cout << "N(" << i << ") = " << N_Function[i] << "\n";
}


// ------------------ AVERAGE PATH LEGNTH ------------------


long jenkins(long x, long seed) {
	unsigned long a, b, c;

	//Set up the internal state 
	a = seed + x;
	b = seed;
	c = 0x9e3779b97f4a7c13L; //the golden ratio; an arbitrary value

	a -= b; a -= c; a ^= (c >> 43);
	b -= c; b -= a; b ^= (a << 9);
	c -= a; c -= b; c ^= (b >> 8);
	a -= b; a -= c; a ^= (c >> 38);
	b -= c; b -= a; b ^= (a << 23);
	c -= a; c -= b; c ^= (b >> 5);
	a -= b; a -= c; a ^= (c >> 35);
	b -= c; b -= a; b ^= (a << 49);
	c -= a; c -= b; c ^= (b >> 11);
	a -= b; a -= c; a ^= (c >> 12);
	b -= c; b -= a; b ^= (a << 18);
	c -= a; c -= b; c ^= (b >> 22);

	return c;
}


int countLeadingZeros(int64_t x) { 
    int total_bits = sizeof(x) * 8;

    int res = 0; 
    while ( !(x & (static_cast<uint64_t>(1) << (total_bits - 1))) ) { 
        x = (x << 1); 
        res++; 
    } 
    return res; 
} 


double computeAlpha(uint64_t m) {
	double alpha;
    switch (m) {
        case 16:
            alpha = 0.673;
            break;
        case 32:
            alpha = 0.697;
            break;
        case 64:
            alpha = 0.709;
            break;
        default:
            alpha = 0.7213 / (1.0 + 1.079 / m);
            break;
    }
    return alpha;
}


void updateCounter(vector< vector<int> >& c, vector< vector<int> >& temp) {
	for(int i = 0; i < temp.size(); i++){
		int v = temp[i][0];
		int n = temp[i][1];
		int value = temp[i][2];
		if (value > c[v][n]){
			c[v][n] = value;
		}
	}
}


void add(vector< vector<int> >& c, long v, int b, long seed) {
	
	int64_t x = jenkins(v, seed);
	int j = static_cast<uint64_t>(x) >> (64-b);
	int w = countLeadingZeros(x << b) + 1;

	c[v][j] = max(w,c[v][j]);
} 


uint64_t size(vector< vector<int> >& c, uint64_t m, double alpha, long v) {
	long double denominator = 0;
	for(uint64_t i=0; i<m; i++){
		denominator += pow(2, - c[v][i]);
	}
	long double z = 1/denominator;
	uint64_t e = alpha * pow(m,2) * z;
	return e;
}


void Union(vector< vector<int> >& c,  long u, long v, uint64_t m, vector< vector<int> >& temp) {
	vector<int> entry;
	for(int i=0; i<m; i++){
		if(c[v][i] > c[u][i]){
			flag = true;
			entry.push_back(u);
			entry.push_back(i);
			entry.push_back(c[v][i]);
			temp.push_back(entry);
		}
	}
}


void computeNeighbourhoodFunction(vector< vector<int> >& c, vector<unsigned long int>& N_Function, uint64_t m, double alpha) {
	uint64_t size_node;
	uint64_t total_size = 0;
	for(uint64_t v = 0; v < N; v++) {
		size_node = size(c, m, alpha, v);
		total_size += size_node;
	}
	N_Function.push_back(total_size);
}


long double computeAverageDistance(int max_distance, vector<unsigned long int>& N_Function) {
	long double average_distance = 0.0;
	for (int i = 0; i < max_distance-1; i++){
		unsigned long int N_difference = N_Function[i+1] - N_Function[i];
		average_distance += N_difference * (i+1);
	}

	average_distance /= N_Function[max_distance-1];
	return average_distance;
}


long double APL(int b) {

	unsigned seed = chrono::system_clock::now().time_since_epoch().count(); 
	uint64_t m = pow(2,b);

	flag = true;
	double alpha = computeAlpha(m);
	vector<vector<int> > c( N, vector<int> (m, 0));
	
	// initialize counters with the nodes id
	for(uint64_t v = 0; v < N; v++) {
		add(c, v, b, seed);
	}

	vector<unsigned long int> N_Function;
	vector<vector<int> > temp;
	int t = 0;
	int d;
	while(flag) {
		flag = false;
		temp.clear();
		vector<vector<int> >(temp).swap(temp);

		// do the Union for each edge in the graph
		for(uint64_t v = 0; v < N; v++){
			vector<int> a = c[v];
			for(int j = offset[v]; j < offset[v+1]; j++){
				int w = neighbour[j];
				Union(c, v, w, m, temp);						// B(v, t+1)
			}
		}
		
		updateCounter(c, temp);
		computeNeighbourhoodFunction(c, N_Function, m, alpha);
		t += 1;
	}

	int max_distance = N_Function.size();
	//printNeighbourhoods(max_distance, N_Function);
	long double avg_distance = computeAverageDistance(max_distance, N_Function);

	return avg_distance;

}


// ------------------ BETWEENNESS ------------------


vector<double> Brandes() {

    vector<double> c(N,0);
    
    for(int s = 0; s < N; s++) 
    {
        // initialization
        queue<int> Q;
        stack<int> S;
        vector<int> distance(N,-1);    distance[s] = 0;
        vector<vector<int> > predecessors(N);
        vector<double> sigma(N,0); sigma[s] = 1; // number of shortest paths from source to v
        
        Q.push(s);
        while(!Q.empty())
        {
            int v = Q.front(); Q.pop();
            S.push(v);
            for (int i = offset[v]; i < offset[v+1]; i++)
            {
                int w = neighbour[i];
                // path discovery
                if (distance[w] < 0)
                {
                    Q.push(w);
                    distance[w] = distance[v] + 1;
                }
                // path counting
                if (distance[w] == distance[v] + 1) 
                {
                    sigma[w] = sigma[w] + sigma[v];
                    predecessors[w].push_back(v);
                }
            }
        }

        // acumulation
        vector<double> delta(N,0); //dependency of source on v
        while(!S.empty())
        {
            int w = S.top(); S.pop();
            for(int i=0; i < predecessors[w].size(); i++) 
            {
                int v = predecessors[w][i];
                delta[v] = delta[v] + sigma[v] / sigma[w] * (1 + delta[w]);
            }
            if (w != s)c[w] = c[w] + delta[w];
        }
    }
    
    for(int i=0; i < N; i++)
        c[i] = c[i] / ((N-1)*(N-2)); // normalization for undirected graph

    return c;
}


//A BFS starting at a random vertex. Returns a 2-approximation of the graph's diameter
int VertexDiameter2Approx() {
	queue<int> q;
	srand(time(NULL));
	int v = rand() % N;
	q.push(v);
	vector<bool> visited(N,false); visited[v] = true;
	vector<int>  distance(N,0);
	int source;  int neighb; 
	int s = 0;   int s1 = 0; int s2 = 0;
	while (!q.empty()) 
	{
		source = q.front(); q.pop();
		for (int i = offset[source]; i < offset[source+1]; i++)
		{
			neighb = neighbour[i];
			if (!visited[neighb]) 
			{
				s = distance[neighb] = distance[source] + 1;
				visited[neighb] = true;
				q.push(neighb);

				//update the two longest paths
				if (s > s1) { s2 = s1; s1 = s; }  
				else if (s > s2) s2 = s;
			}
		}
	}
	return s1+s2;
}


//A BFS adaption for our purposes
pair< vector<vector <int> >, vector<int> > AllShortestPaths(int source, int target) {
	queue<int> q; q.push(source);

	vector<int>  sigma_u(N,0);      sigma_u[source]  = 1;
	vector<int>  distance(N,-1);    distance[source] = 0;
	vector<bool> enqueued(N,false); enqueued[source] = true;
	
	vector<vector<int> > predecessors(N);
	
	int neighb;

	// we're assuming that "target" is always reachable from "source"
	while (q.front() != target)
	{
		source = q.front(); q.pop();
		enqueued[source] = false;

		for (int i = offset[source]; i < offset[source+1]; i++) 
		{
			neighb = neighbour[i];
			if (distance[neighb] == -1) distance[neighb] = distance[source] + 1;
			if (distance[neighb] == distance[source] + 1) 
			{
				predecessors[neighb].push_back(source);
				if (!enqueued[neighb]) 
				{
					q.push(neighb);
					enqueued[neighb] = true;
				}
				sigma_u[neighb] += sigma_u[source];
			}
		}
	}
	pair< vector<vector <int> >, vector<int> > x(predecessors, sigma_u);
	return x;
}


vector<double> Riondato(double c, double eps, double delta) {
	assert(c<=1 && c>=0 && eps<=1 && eps>=0);

	unsigned seed = chrono::system_clock::now().time_since_epoch().count(); 
	default_random_engine generator (seed);
	srand (time(NULL));
	c     = 0.5;
	eps   = 0.5;
	delta = 0.8;
	vector<double> betweenness(N,0);
	int source = 0; 
	int target = 0;
	int diameter = VertexDiameter2Approx();
	if (diameter < 2) return betweenness;
	double r 	 = (c / (eps*eps)) * ( floor(log2(diameter-2)) + log(1/delta));

	
	for (int i = 0; i < r; i++)
	{
		do
		{
			source = rand() % N;
			target = rand() % N;
		} while (source == target);

		pair< vector<vector <int> >, vector<int> > x = AllShortestPaths(source,target);
		vector<vector <int> > predecessors = x.first;
		vector<int> 		  sigma_u      = x.second;

		int t = target; int parents = 0; int z = 0;
		while (t != source)
		{
			parents = predecessors[t].size();
			
			vector<double> a(parents);
			for (int i = 0; i < parents; i++) a[i] = 1.0*sigma_u[predecessors[t][i]]/sigma_u[t];
			discrete_distribution<int> distribution(a.begin(),a.end());
			z = distribution(generator);
			z = predecessors[t][z];

			if (z != source) betweenness[z] = betweenness[z] + 1.0/r;
			
			t = z;
		}
	}

	return betweenness;
}


// ------------------ CLUSTERING ------------------

long double Clustering() {
	long long int triangles=0;

	for (int i = 0; i < N; ++i)
		for (int j = i+1; j < N; ++j)
			for (int k = j+1; k < N; ++k)
				if (IsAdjacent(i,j) && IsAdjacent(j,k) && IsAdjacent(k,i))
					triangles++;
	return 6.0*triangles / (N*(N-1)*(N-2));
}


int BinarySearch(int r, int AccWedgeCount[]) {
	int L = 0;
	int R = N-1;
	int guess;
	int res  = -1;
	int itvl = -1;

	while (L <= R)
	{
		guess = L + (R-L)/2;  // simplified form: (L+R)/2
		itvl  = guess;
		if (AccWedgeCount[guess] == r) 
		{
			res = guess;      // possibly found the wedge we're looking for
			R   = guess - 1;
		}
		else if (AccWedgeCount[guess] < r) { L = guess + 1; itvl++; }
		else if (AccWedgeCount[guess] > r) { R = guess - 1; itvl--; }
	}
	return res != -1 ? res: itvl;
}


long double UniformWedge(int sample_sz) { //sample_sz is the number of wedges we want to sample. play with this parameter

	int TW = 0;
	int AccWedgeCount[N];
	int degree;
	
	for (int i = 0; i < N; i++) 
	{
		AccWedgeCount[i] = TW;
 		degree = offset[i+1]-offset[i];
		TW += degree*(degree-1) / 2;
	}

	long long int sum = 0;
	int center, r, v1, v2, wedge;

	for (int i = 0; i < sample_sz; i++)
	{

		wedge = (rand() % (TW+1)); // "wedge" lives in {0, 1, ..., TW}
		center = BinarySearch(wedge, AccWedgeCount);

		//generate random wedge where node of index "center" is at its center
		degree = offset[center+1] - offset[center];
		if (degree<2) continue;

		v1, v2 = degree;
		do
		{
			v1 = rand() % degree;
			v2 = rand() % degree;
		} while (v1 == v2);

		v1 = neighbour[offset[center]+v1];
		v2 = neighbour[offset[center]+v2];
		sum += IsAdjacent(v1, v2) ? 1 : 0; // c(w)=1 if triangle, 0 otherwise
	}

	return sum / (3.0*sample_sz);
}


double ApproxClusteringNaive(int s) {
	assert(s>3);

	int u; int v1; int v2; int degree; int ct=0;
	for (int i = 0; i < s; i++)
	{
		u  = rand() % N;
		degree = offset[u+1]-offset[u];
		if (degree<2) continue;
		do
		{
			v1 = rand() % degree;
			v2 = rand() % degree;
		} while (v1 == v2);
		v1 = neighbour[offset[u]+v1];
		v2 = neighbour[offset[u]+v2];
		if (IsAdjacent(v1,v2)) ct++;
	}
	return 1.0*ct/s;
}


// ------------------ MAIN ------------------

int main() {
	
	/* N: number of nodes   ||   M: 2 * number of links   */
	cin >> N >> M;

	neighbour = (int*) malloc(sizeof(int)*M); 
	offset    = (int*) malloc(sizeof(int)*(N+1)); // +1 is a hack to know that we are at the end of the array

	int u = 0; int v = 0;
	int previous = 0;
	int shift = 0;
	offset[0] = 0;
	for (int i = 0; i < M; i++) {
		cin >> u >> v;
		if (u == previous) {
			neighbour[i] = v;
			shift++;
		}
		else {
			neighbour[i] = v;
			for (int j = previous; j < u; j++)
				offset[j+1] = shift;
			previous = u;
			shift++;
		}
	}
	offset[N] = M;
	
	cout << "GRAPH IN CSR FORMAT\n";
	printGraph(neighbour, offset);
	cout << "\n";

	cout << "CLUSTERING COEFFICIENT\n";
	cout << Clustering()    << endl;
	cout << UniformWedge(3) << endl;          //random sample size
	cout << ApproxClusteringNaive(4) << endl; //random sample size
	cout << endl;

	cout << "BETWEENNESS CENTRALITY\n";
	vector<double> betweenness_ = Riondato(0,0,0);
	for (int i = 0; i < betweenness_.size(); i++) {
		cout << "Node " << i << " -> " << betweenness_[i] << endl;
	}
	cout << "\n";

	cout << "AVERAGE PATH LENGTH\n";
	cout << APL(4) << endl; //random value of m

	cout << "\nEND" << endl;
}


/*	Modified: 14/03/2021
	SIMPLIFIED CSR ... 
	int u = 0; 
	int v = 0;
	int previous = 0;

	for (int i = 0; i < NUMBER_OF_EDGES; ++i)
	{
		cin >> u >> v;
		adj[i] = v;
		if (u != previous) 
		{
			offset[u] = i;
			previous = u;
		}
	}
	offset[N] = NUMBER_OF_EDGES;
*/
