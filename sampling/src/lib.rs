mod proba_estimator;
use crossterm::{cursor, QueueableCommand};
use numpy::{PyReadonlyArray1, PyReadonlyArray4};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rayon::prelude::*;
use std::io::{stdout, Write};
use std::time::Instant;

#[pymodule]
fn sampling(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "get_mi_modes")]
    fn get_mi_modes<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        matrix: PyReadonlyArray4<f64>,
        std: PyReadonlyArray1<f64>,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let matrix = matrix.as_array();
        let std = std.as_array();
        let n = 1000;
        let h = f64::log2(matrix.shape()[0] as f64);

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_pr_from_modes(&matrix, &std));
            let res: Vec<f64> = par_iter.collect();
            s = res.iter().fold(s, |acc, x| acc + x);
            s2 = res.iter().fold(s2, |acc, x| acc + (x.powi(2)));
            ns += n as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }
    #[pyfn(m, "get_pi_shuffling_mac12")]
    fn get_pi_shuffling_mac12<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 5000;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_pi_shuffling_mac12(std, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }



#[pyfn(m, "get_pi_shuffling_mnew")]
    fn get_pi_shuffling_mnew<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 5000;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_pi_shuffling_mnew(std, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }


    #[pyfn(m, "get_mi_shuffling")]
    fn get_mi_shuffling<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 50;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_mi_shuffling(std, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }

    #[pyfn(m, "get_mi_lin_m_tuples")]
    fn get_mi_lin_m_tuples<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        d: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 5000;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_mi_lin_m_tuples(std, d, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }

    #[pyfn(m, "get_mi_lin_m_shares")]
    fn get_mi_lin_m_shares<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        d: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 5000;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_mi_lin_m_shares(std, d, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }

    #[pyfn(m, "get_mi_lin_m")]
    fn get_mi_lin_m<'py>(
        py: Python<'py>,
        preci: f64,
        timeout: f64,
        eta: usize,
        d: usize,
        std: f64,
    ) -> f64 {
        let mut stdout = stdout();
        let start = Instant::now();

        let n = 1000;
        let n_traces = 5000;
        let h = eta as f64;

        let mut ns = 0.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut std_mi = 1.0;
        let mut mi = 0.0;

        while ((std_mi / mi) > preci) | (mi <= 0.0) {
            let par_iter = (0..n)
                .into_par_iter()
                .map(|_| proba_estimator::sample_mi_lin_m(std, d, eta, n_traces));
            let res: Vec<(f64, f64)> = par_iter.collect();
            let x = res.iter().fold((s, s2), |(s, s2), (x, y)| (s + x, s2 + y));
            s = x.0;
            s2 = x.1;
            ns += (n * n_traces) as f64;

            mi = h + s / ns;
            std_mi = f64::sqrt(((s2 / ns) - (s / ns).powi(2)) / ns);

            let now = Instant::now();
            stdout.queue(cursor::SavePosition).unwrap();
            stdout
                .write(
                    format!(
                        "{:4.3} [sec] --- MI {:.6} --- Preci {:.6} --- Samples {:.3} [M]",
                        now.duration_since(start).as_secs_f64(),
                        mi,
                        std_mi / mi,
                        ns / 1E6
                    )
                    .as_bytes(),
                )
                .unwrap();
            stdout.queue(cursor::RestorePosition).unwrap();
            stdout.flush().unwrap();
            py.check_signals().unwrap();
            if (now.duration_since(start).as_secs_f64() as f64) > timeout {
                return -1.0;
            }
        }
        mi
    }

    Ok(())
}
