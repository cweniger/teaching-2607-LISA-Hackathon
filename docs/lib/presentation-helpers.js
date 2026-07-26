/* Presentation deck — shared numerical helpers. */

function createRNG(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussN(rng) { return Math.sqrt(-2*Math.log(rng()+1e-12))*Math.cos(2*Math.PI*rng()); }

function matVec(A, v) { const m=A.length, n=v.length, r=new Array(m).fill(0); for(let i=0;i<m;i++){let s=0; for(let j=0;j<n;j++) s+=A[i][j]*v[j]; r[i]=s;} return r; }
function solve(A, b) {
  // Gauss elimination with partial pivoting.
  const n = b.length;
  const M = A.map((row,i)=>row.slice().concat(b[i]));
  for (let i=0;i<n;i++){
    let p = i; for (let k=i+1;k<n;k++) if (Math.abs(M[k][i])>Math.abs(M[p][i])) p=k;
    [M[i],M[p]]=[M[p],M[i]];
    const piv = M[i][i] || 1e-12;
    for (let k=i+1;k<n;k++){ const f=M[k][i]/piv; for(let j=i;j<=n;j++) M[k][j] -= f*M[i][j]; }
  }
  const x = new Array(n).fill(0);
  for (let i=n-1;i>=0;i--){ let s=M[i][n]; for (let j=i+1;j<n;j++) s -= M[i][j]*x[j]; x[i] = s/(M[i][i]||1e-12); }
  return x;
}
function polyDesign(xs, M) { return xs.map(x => { const r=new Array(M); let v=1; for(let j=0;j<M;j++){ r[j]=v; v*=x; } return r; }); }
function polyEval(w, x) { let v=1, s=0; for(let j=0;j<w.length;j++){ s+=w[j]*v; v*=x; } return s; }
function fitPoly(xs, ts, M) {
  const Phi = polyDesign(xs, M);
  const N = xs.length;
  const A = []; for(let i=0;i<M;i++){ const row=new Array(M).fill(0); for(let j=0;j<M;j++){ let s=0; for(let n=0;n<N;n++) s+=Phi[n][i]*Phi[n][j]; row[j]=s; } A.push(row); }
  const bv = new Array(M).fill(0); for(let i=0;i<M;i++){ let s=0; for(let n=0;n<N;n++) s+=Phi[n][i]*ts[n]; bv[i]=s; }
  return solve(A, bv);
}
