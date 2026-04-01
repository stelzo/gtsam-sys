#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap, HashSet};

use faer::{
    Mat,
    Side,
    linalg::solvers::Solve,
    sparse::{Pair, SparseColMat, SymbolicSparseColMat, Triplet, linalg::solvers},
};
use gtsam_rs_inference::Key;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub struct Ordering {
    pub keys: Vec<Key>,
}

impl Ordering {
    pub fn natural(keys: Vec<Key>) -> Self {
        Self { keys }
    }
}

pub trait GaussianFactor {
    fn keys(&self) -> &[Key];
    fn jacobian(&self) -> &DMatrix<f64>;
    fn rhs(&self) -> &DVector<f64>;
}

#[derive(Default)]
pub struct GaussianFactorGraph {
    pub factors: Vec<Box<dyn GaussianFactor + Send + Sync>>,
}

impl GaussianFactorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor<F>(&mut self, factor: F)
    where
        F: GaussianFactor + Send + Sync + 'static,
    {
        self.factors.push(Box::new(factor));
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OrderingStrategy {
    Natural,
    Reverse,
    MinDegree,
    MinFill,
    ColamdLike,
}

type Block<const N: usize> = SMatrix<f64, N, N>;
type VecN<const N: usize> = SVector<f64, N>;

#[derive(Clone, Debug)]
pub struct GaussianConditional<const N: usize> {
    pub frontal: usize,
    pub rhs: VecN<N>,
    pub coeffs: Vec<(usize, Block<N>)>,
}

#[derive(Clone, Debug)]
pub struct BayesTreeClique<const N: usize> {
    pub id: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub conditional: GaussianConditional<N>,
}

#[derive(Clone, Debug)]
pub struct GaussianBayesTree<const N: usize> {
    pub roots: Vec<usize>,
    pub cliques: Vec<BayesTreeClique<N>>,
}

#[derive(Clone, Debug)]
pub struct SymbolicBlockElimination {
    pub ordering: Vec<usize>,
    pub neighbors: Vec<Vec<usize>>,
    pub parent: Vec<Option<usize>>,
    pub roots: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct FaerHessianOrder {
    pub sparsity_pattern: SymbolicSparseColMat<usize>,
    pub sparsity_order: faer::sparse::Argsort<usize>,
    symbolic_llt: solvers::SymbolicLlt<usize>,
    entries: Vec<(usize, usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct SparseBlockSystem<const N: usize> {
    nblocks: usize,
    h: HashMap<(usize, usize), Block<N>>,
    b: Vec<VecN<N>>,
}

pub type SparseBlockSystem3 = SparseBlockSystem<3>;
pub type SparseBlockSystem6 = SparseBlockSystem<6>;

impl<const N: usize> SparseBlockSystem<N> {
    pub fn new(nblocks: usize) -> Self {
        Self::with_block_capacity(nblocks, 0)
    }

    pub fn with_block_capacity(nblocks: usize, nnz_blocks_hint: usize) -> Self {
        Self {
            nblocks,
            h: HashMap::with_capacity(nnz_blocks_hint),
            b: vec![VecN::<N>::zeros(); nblocks],
        }
    }

    pub fn nblocks(&self) -> usize {
        self.nblocks
    }

    fn canonical_key(i: usize, j: usize) -> (usize, usize, bool) {
        if i <= j {
            (i, j, false)
        } else {
            (j, i, true)
        }
    }

    pub fn add_h(&mut self, i: usize, j: usize, block: &Block<N>) {
        let (a, b, transposed) = Self::canonical_key(i, j);
        let value = if transposed { block.transpose() } else { *block };
        self.h
            .entry((a, b))
            .and_modify(|v| *v += value)
            .or_insert(value);
    }

    pub fn add_b(&mut self, i: usize, block: VecN<N>) {
        self.b[i] += block;
    }

    pub fn get_h(&self, i: usize, j: usize) -> Block<N> {
        let (a, b, transposed) = Self::canonical_key(i, j);
        if let Some(v) = self.h.get(&(a, b)) {
            if transposed {
                v.transpose()
            } else {
                *v
            }
        } else {
            Block::<N>::zeros()
        }
    }

    pub fn b_block(&self, i: usize) -> VecN<N> {
        self.b[i]
    }

    fn set_h(&mut self, i: usize, j: usize, block: Block<N>) {
        let (a, b, transposed) = Self::canonical_key(i, j);
        let value = if transposed { block.transpose() } else { block };
        if value.norm() < 1e-14 {
            self.h.remove(&(a, b));
        } else {
            self.h.insert((a, b), value);
        }
    }

    pub fn eliminate_to_bayes_tree(
        &self,
        ordering: &[usize],
        lambda: f64,
    ) -> Result<GaussianBayesTree<N>, String> {
        let symbolic = symbolic_from_h_pattern(self.nblocks, self.h.keys().copied(), ordering)?;
        self.eliminate_to_bayes_tree_with_symbolic(&symbolic, lambda)
    }

    pub fn eliminate_to_bayes_tree_with_symbolic(
        &self,
        symbolic: &SymbolicBlockElimination,
        lambda: f64,
    ) -> Result<GaussianBayesTree<N>, String> {
        if symbolic.ordering.len() != self.nblocks {
            return Err("symbolic ordering size does not match system size".to_string());
        }

        let mut work = self.clone();
        for i in 0..work.nblocks {
            let d = work.get_h(i, i) + Block::<N>::identity() * lambda;
            work.set_h(i, i, d);
        }

        let mut cond: Vec<Option<GaussianConditional<N>>> = vec![None; work.nblocks];

        for &k in &symbolic.ordering {
            let hkk = work.get_h(k, k);
            let inv_hkk = if let Some(inv) = hkk.try_inverse() {
                inv
            } else {
                let mut inv_opt = None;
                let mut jitter = 1e-12;
                for _ in 0..8 {
                    let hkkr = hkk + Block::<N>::identity() * jitter;
                    if let Some(inv) = hkkr.try_inverse() {
                        inv_opt = Some(inv);
                        break;
                    }
                    jitter *= 10.0;
                }
                inv_opt.ok_or_else(|| format!("singular pivot at block {}", k))?
            };
            let bk = work.b[k];
            let inv_bk = inv_hkk * bk;
            let rhs = -inv_bk;

            let neighbors: &[usize] = &symbolic.neighbors[k];
            let mut coeffs = Vec::with_capacity(neighbors.len());
            for &j in neighbors {
                let hkj = work.get_h(k, j);
                coeffs.push((j, inv_hkk * hkj));
            }

            for &i in neighbors {
                let hik = work.get_h(i, k);
                work.b[i] -= hik * inv_bk;
            }

            for (idx_i, &i) in neighbors.iter().enumerate() {
                let hik = work.get_h(i, k);
                for &j in neighbors.iter().skip(idx_i) {
                    let hkj = work.get_h(k, j);
                    let hij = work.get_h(i, j) - hik * inv_hkk * hkj;
                    work.set_h(i, j, hij);
                }
            }

            work.h.remove(&(k, k));
            for &j in neighbors {
                let (a, b, _) = Self::canonical_key(k, j);
                work.h.remove(&(a, b));
            }
            cond[k] = Some(GaussianConditional {
                frontal: k,
                rhs,
                coeffs,
            });
        }

        let mut cliques = Vec::with_capacity(self.nblocks);
        let mut clique_of_var = vec![usize::MAX; self.nblocks];
        for &v in &symbolic.ordering {
            let conditional = cond[v]
                .take()
                .ok_or_else(|| format!("missing conditional for block {}", v))?;
            let id = cliques.len();
            clique_of_var[v] = id;
            cliques.push(BayesTreeClique {
                id,
                parent: None,
                children: Vec::new(),
                conditional,
            });
        }

        for clique in &mut cliques {
            let frontal = clique.conditional.frontal;
            clique.parent = symbolic.parent[frontal].map(|v| clique_of_var[v]);
        }

        let mut roots = Vec::new();
        let parents: Vec<Option<usize>> = cliques.iter().map(|c| c.parent).collect();
        for (child_id, parent_opt) in parents.into_iter().enumerate() {
            match parent_opt {
                Some(parent_id) => cliques[parent_id].children.push(child_id),
                None => roots.push(child_id),
            }
        }

        Ok(GaussianBayesTree { roots, cliques })
    }

    pub fn solve_with_bayes_tree(
        &self,
        ordering: &[usize],
        lambda: f64,
    ) -> Result<DVector<f64>, String> {
        let tree = self.eliminate_to_bayes_tree(ordering, lambda)?;
        tree.solve(self.nblocks)
    }

    pub fn solve_with_symbolic(
        &self,
        symbolic: &SymbolicBlockElimination,
        lambda: f64,
    ) -> Result<DVector<f64>, String> {
        match self.eliminate_to_bayes_tree_with_symbolic(symbolic, lambda) {
            Ok(tree) => tree.solve(self.nblocks),
            Err(_) => self.solve_with_faer_llt(lambda),
        }
    }

    pub fn solve(&self, ordering: &[usize], lambda: f64) -> Result<DVector<f64>, String> {
        match self.solve_with_bayes_tree(ordering, lambda) {
            Ok(x) => Ok(x),
            Err(_) => self.solve_with_faer_llt(lambda),
        }
    }

    pub fn solve_with_faer_order(
        &self,
        faer_order: &FaerHessianOrder,
        lambda: f64,
    ) -> Result<DVector<f64>, String> {
        let dim = self.nblocks * N;
        let mut values = Vec::<f64>::with_capacity(faer_order.entries.len());

        for &(row_block, col_block, local_r, local_c) in &faer_order.entries {
            let row = row_block * N + local_r;
            let col = col_block * N + local_c;
            let block = self.get_h(row_block, col_block);
            let mut value = block[(local_r, local_c)];
            if row == col {
                value += lambda;
            }
            values.push(value);
        }

        let a = SparseColMat::<usize, f64>::new_from_argsort(
            faer_order.sparsity_pattern.clone(),
            &faer_order.sparsity_order,
            &values,
        )
        .map_err(|e| format!("faer sparse matrix build failed: {e:?}"))?;

        let llt =
            solvers::Llt::try_new_with_symbolic(faer_order.symbolic_llt.clone(), a.as_ref(), Side::Lower)
            .map_err(|e| format!("faer numeric LLT failed: {e:?}"))?;

        let mut rhs = Mat::<f64>::zeros(dim, 1);
        for i in 0..self.nblocks {
            for r in 0..N {
                rhs[(i * N + r, 0)] = -self.b[i][r];
            }
        }
        let x = llt.solve(&rhs);

        let mut out = DVector::<f64>::zeros(dim);
        for i in 0..dim {
            out[i] = x[(i, 0)];
        }
        Ok(out)
    }

    pub fn solve_with_faer_llt(&self, lambda: f64) -> Result<DVector<f64>, String> {
        let dim = self.nblocks * N;
        let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
        triplets.reserve(self.h.len() * N * N);

        for (&(i, j), block) in &self.h {
            if i == j {
                for r in 0..N {
                    for c in 0..=r {
                        let mut value = block[(r, c)];
                        if r == c {
                            value += lambda;
                        }
                        if value.abs() > 1e-14 {
                            triplets.push(Triplet::new(i * N + r, j * N + c, value));
                        }
                    }
                }
            } else {
                for r in 0..N {
                    for c in 0..N {
                        let value = block[(c, r)];
                        if value.abs() > 1e-14 {
                            triplets.push(Triplet::new(j * N + r, i * N + c, value));
                        }
                    }
                }
            }
        }

        // Keep completely empty diagonal rows numerically anchored.
        for i in 0..dim {
            triplets.push(Triplet::new(i, i, 0.0));
        }

        let a = SparseColMat::<usize, f64>::try_new_from_triplets(dim, dim, &triplets)
            .map_err(|e| format!("faer sparse matrix build failed: {e:?}"))?;

        let symbolic = solvers::SymbolicLlt::try_new(a.symbolic(), Side::Lower)
            .map_err(|e| format!("faer symbolic LLT failed: {e:?}"))?;
        let llt = solvers::Llt::try_new_with_symbolic(symbolic, a.as_ref(), Side::Lower)
            .map_err(|e| format!("faer numeric LLT failed: {e:?}"))?;

        let mut rhs = Mat::<f64>::zeros(dim, 1);
        for i in 0..self.nblocks {
            for r in 0..N {
                rhs[(i * N + r, 0)] = -self.b[i][r];
            }
        }
        let x = llt.solve(&rhs);

        let mut out = DVector::<f64>::zeros(dim);
        for i in 0..dim {
            out[i] = x[(i, 0)];
        }
        Ok(out)
    }
}

impl<const N: usize> GaussianBayesTree<N> {
    pub fn solve(&self, nblocks: usize) -> Result<DVector<f64>, String> {
        let mut x = vec![VecN::<N>::zeros(); nblocks];
        for &root in &self.roots {
            self.solve_subtree(root, &mut x)?;
        }

        let mut out = DVector::<f64>::zeros(nblocks * N);
        for i in 0..nblocks {
            for r in 0..N {
                out[i * N + r] = x[i][r];
            }
        }
        Ok(out)
    }

    fn solve_subtree(&self, clique_id: usize, x: &mut [VecN<N>]) -> Result<(), String> {
        let clique = self
            .cliques
            .get(clique_id)
            .ok_or_else(|| format!("invalid clique id {}", clique_id))?;
        let frontal = clique.conditional.frontal;
        let mut val = clique.conditional.rhs;
        for (j, a_fj) in &clique.conditional.coeffs {
            val -= *a_fj * x[*j];
        }
        x[frontal] = val;

        for &child in &clique.children {
            self.solve_subtree(child, x)?;
        }
        Ok(())
    }
}

pub fn compute_ordering_from_edges(
    n: usize,
    edges: &[(usize, usize)],
    strategy: OrderingStrategy,
) -> Vec<usize> {
    match strategy {
        OrderingStrategy::Natural => (0..n).collect(),
        OrderingStrategy::Reverse => (0..n).rev().collect(),
        OrderingStrategy::MinDegree => {
            let mut adj = build_adjacency(n, edges);
            eliminate_with_heuristic(&mut adj, |adj, active, i| {
                adj[i].iter().filter(|&&j| active[j]).count()
            })
        }
        OrderingStrategy::MinFill => {
            let mut adj = build_adjacency(n, edges);
            eliminate_with_heuristic(&mut adj, |adj, active, i| {
                let neigh: Vec<usize> = adj[i].iter().copied().filter(|&j| active[j]).collect();
                let mut missing = 0usize;
                for (a_idx, &a) in neigh.iter().enumerate() {
                    for &b in neigh.iter().skip(a_idx + 1) {
                        if !adj[a].contains(&b) {
                            missing += 1;
                        }
                    }
                }
                missing
            })
        }
        OrderingStrategy::ColamdLike => {
            // Cheap COLAMD-like proxy: minimize fill first, then degree.
            let mut adj = build_adjacency(n, edges);
            eliminate_with_heuristic(&mut adj, |adj, active, i| {
                let neigh: Vec<usize> = adj[i].iter().copied().filter(|&j| active[j]).collect();
                let degree = neigh.len();
                let mut missing = 0usize;
                for (a_idx, &a) in neigh.iter().enumerate() {
                    for &b in neigh.iter().skip(a_idx + 1) {
                        if !adj[a].contains(&b) {
                            missing += 1;
                        }
                    }
                }
                missing.saturating_mul(1024) + degree
            })
        }
    }
}

pub fn symbolic_from_edges(
    n: usize,
    edges: &[(usize, usize)],
    ordering: &[usize],
) -> Result<SymbolicBlockElimination, String> {
    symbolic_from_h_pattern(
        n,
        edges
            .iter()
            .copied()
            .flat_map(|(i, j)| if i == j { vec![(i, i)] } else { vec![(i, j), (i, i), (j, j)] }),
        ordering,
    )
}

fn symbolic_from_h_pattern<I>(
    n: usize,
    h_keys: I,
    ordering: &[usize],
) -> Result<SymbolicBlockElimination, String>
where
    I: Iterator<Item = (usize, usize)>,
{
    if ordering.len() != n {
        return Err("ordering size does not match symbolic size".to_string());
    }

    let mut pos = vec![usize::MAX; n];
    for (rank, &v) in ordering.iter().enumerate() {
        if v >= n {
            return Err(format!("ordering contains out-of-range block {}", v));
        }
        if pos[v] != usize::MAX {
            return Err(format!("ordering duplicates block {}", v));
        }
        pos[v] = rank;
    }
    for (v, p) in pos.iter().enumerate() {
        if *p == usize::MAX {
            return Err(format!("ordering missing block {}", v));
        }
    }

    let mut adj = vec![HashSet::<usize>::new(); n];
    for (i, j) in h_keys {
        if i < n && j < n && i != j {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }

    let mut neighbors = vec![Vec::<usize>::new(); n];
    let mut parent = vec![None; n];
    let mut active = vec![true; n];

    for &k in ordering {
        let mut neigh: Vec<usize> = adj[k].iter().copied().filter(|&j| active[j]).collect();
        neigh.sort_unstable_by_key(|j| pos[*j]);
        neighbors[k] = neigh.clone();
        parent[k] = neigh.first().copied();

        for (a_idx, &a) in neigh.iter().enumerate() {
            for &b in neigh.iter().skip(a_idx + 1) {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        for &j in &neigh {
            adj[j].remove(&k);
        }
        adj[k].clear();
        active[k] = false;
    }

    let mut roots = Vec::new();
    for &v in ordering {
        if parent[v].is_none() {
            roots.push(v);
        }
    }

    Ok(SymbolicBlockElimination {
        ordering: ordering.to_vec(),
        neighbors,
        parent,
        roots,
    })
}

pub fn faer_hessian_order_from_edges<const N: usize>(
    nblocks: usize,
    edges: &[(usize, usize)],
) -> Result<FaerHessianOrder, String> {
    let dim = nblocks * N;
    let mut edge_set = HashSet::<(usize, usize)>::with_capacity(edges.len() * 2);
    for &(i, j) in edges {
        if i < nblocks && j < nblocks && i != j {
            edge_set.insert((i, j));
            edge_set.insert((j, i));
        }
    }

    let mut indices = Vec::<Pair<usize, usize>>::new();
    let mut entries = Vec::<(usize, usize, usize, usize)>::new();
    for col_block in 0..nblocks {
        for local_c in 0..N {
            for row_block in col_block..nblocks {
                let connected = row_block == col_block || edge_set.contains(&(row_block, col_block));
                if !connected {
                    continue;
                }
                for local_r in 0..N {
                    let row = row_block * N + local_r;
                    let col = col_block * N + local_c;
                    if row >= col {
                        indices.push(Pair::new(row, col));
                        entries.push((row_block, col_block, local_r, local_c));
                    }
                }
            }
        }
    }

    let (sparsity_pattern, sparsity_order) =
        SymbolicSparseColMat::try_new_from_indices(dim, dim, &indices)
            .map_err(|e| format!("faer symbolic sparsity build failed: {e:?}"))?;

    let symbolic_llt = solvers::SymbolicLlt::try_new(sparsity_pattern.as_ref(), Side::Lower)
        .map_err(|e| format!("faer symbolic LLT failed: {e:?}"))?;

    Ok(FaerHessianOrder {
        sparsity_pattern,
        sparsity_order,
        symbolic_llt,
        entries,
    })
}

fn build_adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<BTreeSet<usize>> {
    let mut adj = vec![BTreeSet::<usize>::new(); n];
    for &(i, j) in edges {
        if i < n && j < n && i != j {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }
    adj
}

fn eliminate_with_heuristic<F>(adj: &mut [BTreeSet<usize>], cost: F) -> Vec<usize>
where
    F: Fn(&[BTreeSet<usize>], &[bool], usize) -> usize,
{
    let n = adj.len();
    let mut active = vec![true; n];
    let mut order = Vec::with_capacity(n);

    for _ in 0..n {
        let k = (0..n)
            .filter(|&i| active[i])
            .min_by_key(|&i| cost(adj, &active, i))
            .expect("there must be an active node");

        let neigh: Vec<usize> = adj[k].iter().copied().filter(|&j| active[j]).collect();
        for (a_idx, &a) in neigh.iter().enumerate() {
            for &b in neigh.iter().skip(a_idx + 1) {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        for &j in &neigh {
            adj[j].remove(&k);
        }
        active[k] = false;
        order.push(k);
    }

    order
}

#[cfg(test)]
mod tests {
    use super::{
        compute_ordering_from_edges, faer_hessian_order_from_edges, OrderingStrategy,
        SparseBlockSystem3, SparseBlockSystem6,
    };
    use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3};

    #[test]
    fn ordering_returns_permutation() {
        let ord = compute_ordering_from_edges(
            4,
            &[(0, 1), (1, 2), (1, 3)],
            OrderingStrategy::MinFill,
        );
        let mut sorted = ord.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn sparse3_solve_matches_dense_reference() {
        let mut sys = SparseBlockSystem3::new(3);
        let eye = Matrix3::<f64>::identity();
        sys.add_h(0, 0, &(2.0 * eye));
        sys.add_h(1, 1, &(3.0 * eye));
        sys.add_h(2, 2, &(4.0 * eye));
        sys.add_h(0, 1, &(0.2 * eye));
        sys.add_h(1, 2, &(-0.1 * eye));
        sys.add_b(0, Vector3::new(1.0, -2.0, 0.5));
        sys.add_b(1, Vector3::new(-0.2, 0.3, 0.1));
        sys.add_b(2, Vector3::new(0.7, 0.2, -0.8));

        let ord = compute_ordering_from_edges(3, &[(0, 1), (1, 2)], OrderingStrategy::MinDegree);
        let xs = sys.solve(&ord, 1e-9).expect("sparse solve");

        let mut h = DMatrix::<f64>::zeros(9, 9);
        let mut b = DVector::<f64>::zeros(9);
        for i in 0..3 {
            b.rows_mut(i * 3, 3).copy_from(&sys.b_block(i));
            for j in i..3 {
                let blk = sys.get_h(i, j);
                for r in 0..3 {
                    for c in 0..3 {
                        h[(i * 3 + r, j * 3 + c)] = blk[(r, c)];
                        h[(j * 3 + c, i * 3 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..9 {
            h[(i, i)] += 1e-9;
        }
        let xd = h.lu().solve(&(-b)).expect("dense solve");
        assert!((xs - xd).norm() < 1e-7);
    }

    #[test]
    fn sparse3_cached_faer_solve_matches_dense_reference() {
        let mut sys = SparseBlockSystem3::new(3);
        let eye = Matrix3::<f64>::identity();
        sys.add_h(0, 0, &(2.0 * eye));
        sys.add_h(1, 1, &(3.0 * eye));
        sys.add_h(2, 2, &(4.0 * eye));
        sys.add_h(0, 1, &(0.2 * eye));
        sys.add_h(1, 2, &(-0.1 * eye));
        sys.add_b(0, Vector3::new(1.0, -2.0, 0.5));
        sys.add_b(1, Vector3::new(-0.2, 0.3, 0.1));
        sys.add_b(2, Vector3::new(0.7, 0.2, -0.8));

        let faer_order =
            faer_hessian_order_from_edges::<3>(3, &[(0, 1), (1, 2)]).expect("faer order");
        let xs = sys
            .solve_with_faer_order(&faer_order, 1e-9)
            .expect("cached faer solve");

        let mut h = DMatrix::<f64>::zeros(9, 9);
        let mut b = DVector::<f64>::zeros(9);
        for i in 0..3 {
            b.rows_mut(i * 3, 3).copy_from(&sys.b_block(i));
            for j in i..3 {
                let blk = sys.get_h(i, j);
                for r in 0..3 {
                    for c in 0..3 {
                        h[(i * 3 + r, j * 3 + c)] = blk[(r, c)];
                        h[(j * 3 + c, i * 3 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..9 {
            h[(i, i)] += 1e-9;
        }
        let xd = h.lu().solve(&(-b)).expect("dense solve");
        assert!((xs - xd).norm() < 1e-7);
    }

    #[test]
    fn elimination_builds_bayes_tree_topology() {
        let mut sys = SparseBlockSystem3::new(4);
        let eye = Matrix3::<f64>::identity();
        for i in 0..4 {
            sys.add_h(i, i, &(2.0 * eye));
            sys.add_b(i, Vector3::new(0.1, -0.2, 0.05));
        }
        sys.add_h(0, 1, &(0.1 * eye));
        sys.add_h(1, 2, &(0.1 * eye));
        sys.add_h(2, 3, &(0.1 * eye));
        let ordering = vec![0usize, 1usize, 2usize, 3usize];
        let tree = sys
            .eliminate_to_bayes_tree(&ordering, 1e-9)
            .expect("eliminate to tree");

        assert_eq!(tree.cliques.len(), 4);
        assert_eq!(tree.roots.len(), 1);
        assert_eq!(tree.cliques[0].parent, Some(1));
        assert_eq!(tree.cliques[1].parent, Some(2));
        assert_eq!(tree.cliques[2].parent, Some(3));
        assert_eq!(tree.cliques[3].parent, None);
    }

    #[test]
    fn sparse6_solve_matches_dense_reference() {
        let mut sys = SparseBlockSystem6::new(2);
        let eye = SMatrix::<f64, 6, 6>::identity();
        let cross = 0.1 * eye;
        sys.add_h(0, 0, &(3.0 * eye));
        sys.add_h(1, 1, &(2.0 * eye));
        sys.add_h(0, 1, &cross);
        sys.add_b(0, SVector::<f64, 6>::new(1.0, -1.0, 0.5, 0.1, -0.2, 0.3));
        sys.add_b(1, SVector::<f64, 6>::new(-0.4, 0.6, 0.2, -0.1, 0.3, -0.2));

        let ord = vec![0usize, 1usize];
        let xs = sys.solve(&ord, 1e-9).expect("sparse solve");

        let mut h = DMatrix::<f64>::zeros(12, 12);
        let mut b = DVector::<f64>::zeros(12);
        for i in 0..2 {
            b.rows_mut(i * 6, 6).copy_from(&sys.b_block(i));
            for j in i..2 {
                let blk = sys.get_h(i, j);
                for r in 0..6 {
                    for c in 0..6 {
                        h[(i * 6 + r, j * 6 + c)] = blk[(r, c)];
                        h[(j * 6 + c, i * 6 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..12 {
            h[(i, i)] += 1e-9;
        }
        let xd = h.lu().solve(&(-b)).expect("dense solve");
        assert!((xs - xd).norm() < 1e-7);
    }

    #[test]
    #[ignore = "benchmark gate; run explicitly in CI/perf jobs"]
    fn benchmark_gate_sparse_vs_dense_chain() {
        use std::time::Instant;

        let n = 100;
        let mut sys = SparseBlockSystem3::new(n);
        let eye = Matrix3::<f64>::identity();
        for i in 0..n {
            sys.add_h(i, i, &(2.0 * eye));
            sys.add_b(i, Vector3::new(1.0, -1.0, 0.5));
            if i + 1 < n {
                sys.add_h(i, i + 1, &(0.05 * eye));
            }
        }

        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let ord = compute_ordering_from_edges(n, &edges, OrderingStrategy::MinFill);

        let t0 = Instant::now();
        let _ = sys.solve(&ord, 1e-9).expect("sparse solve");
        let sparse_t = t0.elapsed();

        let mut h = DMatrix::<f64>::zeros(n * 3, n * 3);
        let mut b = DVector::<f64>::zeros(n * 3);
        for i in 0..n {
            b.rows_mut(i * 3, 3).copy_from(&sys.b_block(i));
            for j in i..n {
                let blk = sys.get_h(i, j);
                if blk.norm() == 0.0 {
                    continue;
                }
                for r in 0..3 {
                    for c in 0..3 {
                        h[(i * 3 + r, j * 3 + c)] = blk[(r, c)];
                        h[(j * 3 + c, i * 3 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..(n * 3) {
            h[(i, i)] += 1e-9;
        }

        let t1 = Instant::now();
        let _ = h.lu().solve(&(-b)).expect("dense solve");
        let dense_t = t1.elapsed();

        assert!(sparse_t <= dense_t * 2, "sparse={sparse_t:?} dense={dense_t:?}");
    }
}
