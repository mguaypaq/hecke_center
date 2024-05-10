# Multiplication tables for centers of Hecke algebras

Here is some code that I wrote to generate some multiplication tables for the centers of Iwahori-Hecke algebras of type A, for the Geck-Rouquier basis. See:

> Geck, M., Rouquier, R. (1997). Centers and Simple Modules for Iwahori-Hecke Algebras. In: Cabanes, M. (eds) Finite Reductive Groups: Related Structures and Representations. Progress in Mathematics, vol 141. Birkhäuser Boston. <https://doi.org/10.1007/978-1-4612-4124-9_9>

The normalization used is:

$$ (T - q)(T + 1) = 0 $$

The result of running the following command, which took ~1 hour on my laptop, can be found in the file `geck_rouquier_mult.json`:

```sh
./main.py 1 2 3 4 5 6 7
```

This was sufficient for my needs, but the code could certainly be optimized further, and parts of the computation could easily be parallelized.

The code relies on a certain set partition of the symmetric group (see the file `geck_rouquier.py`). Here is the number of blocks in this set partition, for a few values of $n$:

|  `n` | `len(blocks)` |
| ---: | ------------: |
|   1  |             1 |
|   2  |             2 |
|   3  |             4 |
|   4  |            11 |
|   5  |            28 |
|   6  |            91 |
|   7  |           305 |
|   8  |          1245 |
|   9  |          5581 |
|  10  |         29717 |
