Generate large sparse instance with [generate_sparse_qubo](./generate_sparse_qubo.exe) and test with [d_flip_sparse](./d_flip_sparse.exe)<br>
<br>
One million variables, ten million unique off-diagonal edges, and one million diagonal records:<br>
./generate_sparse_qubo large.txt 1000000 10000000 --seed 123456
<br>
./d_flip_sparse large.txt large-results.csv --qubo --csv --time-limit 300 --seed 123456
<br>
<br>
Generate dense instance with [generate_dense_qubo](./generate_dense_qubo.exe) and test with [d_flip_dense](./d_flip_dense.exe)<br>
<br>
A moderate dense instance:<br>
./generate_dense_qubo dense-5000.txt 5000 --seed 42 --min-coeff -100 --max-coeff 100
<br>
Run it as an upper-triangular minimization QUBO:<br>
./d_flip_dense dense-5000.txt results.txt --qubo --time-limit 60 --seed 42
<br>
Generate several instances:<br>
./generate_dense_qubo dense-multiple.txt 3000 --instances 5 --seed 1234

