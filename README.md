Aquila-Genetic Optimization with Shuffle Windows for the Latin Hypercube Design Problem
This repository contains the implementation of the Aquila Optimization-Genetic Algorithm (AOGA), an innovative optimization technique designed to solve high-dimensional Latin Hypercube Design (LHD) problems efficiently. The algorithm introduces novel mechanisms such as Shuffle Window (SW) and Crossover Window (CW) operators to enhance optimization stability and convergence while ensuring structural integrity.

> Manuscript Status:  
> The manuscript is currently under revision in the Arabian Journal for Science and Engineering.

 Keywords: Aquila Optimization, Genetic Algorithm, Latin Hypercube Design (LHD), Shuffle Window, Space-Filling Designs.

Features
- Hybrid optimization combining AO and GA techniques.
- Novel operators: Shuffle Window (SW) & Crossover Window (CW).
- Adaptive rates for improved exploration-exploitation balance.
- Robust performance for large-scale design problems (e.g., 15×50 and 18×80 LHDs).

Installation
This project requires Python and the following dependencies:
 `numpy`,  `random`,  `logging`, `scipy`, `matplotlib`
Usage
To execute the algorithm, run the main file:
```bash
python main.py
```
No Dataset Required
This implementation does not require a specific dataset. All necessary data are generated during runtime.

Contributing
We welcome contributions to improve the project! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

Citation
If you use this code in your research, consider citing the associated manuscript:
```
@article{AOGA2025,
  title={Aquila-Genetic Optimization with Shuffle Windows for the Latin Hypercube Design Problem},
  author={M. Masoudian, R. Hajian, S. H. Erfani},
  journal={Arabian Journal for Science and Engineering},
  year={2025},
  note={Under revision}
}
```

License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Contact
For questions or feedback, please contact:
- R. Hajian (Email: rh.hajian@gmail.com)

