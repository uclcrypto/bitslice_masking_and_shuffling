use itertools::Itertools;
use ndarray::s;
use ndarray::Axis;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView4, Zip};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::Rng;

#[inline(always)]
fn fwht(a: &mut [f64], len: usize) {
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

pub fn sample_pi_shuffling_mac12(std: f64, eta: usize, n_traces: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();

    //let snr_data = Array::range(0.0, 2.0, 1.0).var_axis(Axis(0), 0.0) / std.powi(2);
    //let var_perm = Array::range(0.0, eta as f64, 1.0).var_axis(Axis(0), 0.0) / snr_data;
    let std_perm = std; //f64::sqrt(var_perm.sum());

    // generate all the permutations
    let items = 0..eta;
    let mut all_perms = Vec::new();
    for perm in items.permutations(eta) {
        all_perms.push(perm);
    }
    let nperm = all_perms.len();

    // generate the shares and add noise
    let mut data = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));
    let mut perms = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));

    let secrets = data.fold_axis(Axis(1), 0, |acc, y| (acc << 1) + y);

    // shuffling
    let mut tmp = Array1::<u32>::zeros(eta);
    let tmp = tmp.as_slice_mut().unwrap();
    for (mut x, mut perm) in data.genrows_mut().into_iter().zip(perms.genrows_mut()) {
        let x = x.as_slice_mut().unwrap();
        let perm = perm.as_slice_mut().unwrap();
        for i in 0..eta {
            tmp[i] = x[i];
        }
        let perm_x = &all_perms[rng.gen_range(0..nperm)];
        for i in 0..eta {
            x[i] = tmp[perm_x[i]];
            perm[i] = perm_x[i] as u32;
        }
    }
    // add noise to permutations and data
    let mut leakage = Array2::<f64>::random(data.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&data, |leakage, data| {
        *leakage = *data as f64 + std * *leakage
    });
    let mut leakage_perm = Array2::<f64>::random(perms.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage_perm.zip_mut_with(&perms, |leakage_perm, perms| {
        *leakage_perm = *perms as f64 + std_perm * *leakage_perm
    });

    // full entropy of the secret
    let h = 2usize.pow(eta as u32);

    // storage vectors
    let mut f_l_x = Array3::<f64>::zeros((n_traces, eta, 2)); // f_l_x
    let mut f_l_perm = Array3::<f64>::zeros((n_traces, eta, eta)); // f_l_x
    let mut f_l_s = Array1::<f64>::zeros(h);
    let f_l_s = f_l_s.as_slice_mut().unwrap();

    // compute gaussian distribution for each variable (data and perm)
    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }
    for (i, mut f_l_i) in f_l_perm.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage_perm, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std_perm).powi(2)))
        });
    }

    // apply Bayes' theorem to all intermediate variables
    for mut distri in f_l_perm.lanes_mut(Axis(1)) {
        distri /= distri.sum();
    }

    // MI estimators
    let mut m = 0.0;
    let mut m2 = 0.0;
    // for each trace
    for ((f_l_x, f_l_perm), secret) in f_l_x
        .outer_iter()
        .zip(f_l_perm.outer_iter())
        .zip(secrets.iter())
    {
        for i in 0..h {
            f_l_s[i] = 1.0
        }
        for s in 0..eta {
            let mut pr_0 = 0.0;
            let mut pr_1 = 0.0;
            for (f_l_x, f_l_perm) in f_l_x.outer_iter().zip(f_l_perm.outer_iter()) {
                let f_l_x = f_l_x.as_slice().unwrap();
                let f_l_perm = f_l_perm.as_slice().unwrap();
                pr_0 += f_l_perm[s] * f_l_x[0];
                pr_1 += f_l_perm[s] * f_l_x[1];
            }
            for i in 0..h {
                if (i >> (eta - 1 - s) & 0x1) == 0 {
                    f_l_s[i] *= pr_0;
                } else {
                    f_l_s[i] *= pr_1;
                }
            }
        }
        // Bayes' theorem
        let l = f64::log2(f_l_s[*secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));
        // update mi estimators
        m += l;
        m2 += l.powi(2);
    }

    (m, m2)
}

pub fn sample_pi_shuffling_mnew(std: f64, eta: usize, n_traces: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();

    //let snr_data = Array::range(0.0, 2.0, 1.0).var_axis(Axis(0), 0.0) / std.powi(2);
    //let var_perm = Array::range(0.0, eta as f64, 1.0).var_axis(Axis(0), 0.0) / snr_data;
    let std_perm = std; //f64::sqrt(var_perm.sum());


    // generate all the permutations
    let items = 0..eta;
    let mut all_perms = Vec::new();
    for perm in items.permutations(eta) {
        all_perms.push(perm);
    }
    let nperm = all_perms.len();

    // generate the shares and add noise
    let mut data = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));
    let mut perms = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));

    let secrets = data.fold_axis(Axis(1), 0, |acc, y| (acc << 1) + y);

    // shuffling
    let mut tmp = Array1::<u32>::zeros(eta);
    let tmp = tmp.as_slice_mut().unwrap();
    for (mut x, mut perm) in data.genrows_mut().into_iter().zip(perms.genrows_mut()) {
        let x = x.as_slice_mut().unwrap();
        let perm = perm.as_slice_mut().unwrap();
        for i in 0..eta {
            tmp[i] = x[i];
        }
        let perm_x = &all_perms[rng.gen_range(0..nperm)];
        for i in 0..eta {
            x[i] = tmp[perm_x[i]];
            perm[i] = perm_x[i] as u32;
        }
    }
    // add noise to permutations and data
    let mut leakage = Array2::<f64>::random(data.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&data, |leakage, data| {
        *leakage = *data as f64 + std * *leakage
    });
    let mut leakage_perm = Array2::<f64>::random(perms.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage_perm.zip_mut_with(&perms, |leakage_perm, perms| {
        *leakage_perm = *perms as f64 + std_perm * *leakage_perm
    });

    // full entropy of the secret
    let h = 2usize.pow(eta as u32);

    // storage vectors
    let mut f_l_x = Array3::<f64>::zeros((n_traces, eta, 2)); // f_l_x
    let mut f_l_perm = Array3::<f64>::zeros((n_traces, eta, eta)); // f_l_x
    let mut f_l_s = Array1::<f64>::zeros(h);
    let f_l_s = f_l_s.as_slice_mut().unwrap();

    // compute gaussian distribution for each variable (data and perm)
    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }
    for (i, mut f_l_i) in f_l_perm.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage_perm, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std_perm).powi(2)))
        });
    }

    // apply Bayes' theorem to all intermediate variables
    for mut distri in f_l_perm.genrows_mut() {
        distri /= distri.sum();
    }
    for mut distri in f_l_x.genrows_mut() {
        distri /= distri.sum();
    }

    // MI estimators
    let mut m = 0.0;
    let mut m2 = 0.0;
    // for each trace
    for ((f_l_x, f_l_perm), secret) in f_l_x
        .outer_iter()
        .zip(f_l_perm.outer_iter())
        .zip(secrets.iter())
    {
        for i in 0..h {
            f_l_s[i] = 1.0
        }
        for s in 0..eta {
            let mut pr_0 = 0.0;
            let mut pr_1 = 0.0;
            for (f_l_x, f_l_perm) in f_l_x.outer_iter().zip(f_l_perm.outer_iter()) {
                let f_l_x = f_l_x.as_slice().unwrap();
                let f_l_perm = f_l_perm.as_slice().unwrap();
                pr_0 += f_l_perm[s] * f_l_x[0];
                pr_1 += f_l_perm[s] * f_l_x[1];
            }
            for i in 0..h {
                if (i >> (eta - 1 - s) & 0x1) == 0 {
                    f_l_s[i] *= pr_0;
                } else {
                    f_l_s[i] *= pr_1;
                }
            }
        }
        // Bayes' theorem
        let l = f64::log2(f_l_s[*secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));
        // update mi estimators
        m += l;
        m2 += l.powi(2);
    }

    (m, m2)
}

pub fn sample_mi_shuffling(std: f64, eta: usize, n_traces: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    //let snr_data = Array::range(0.0, 2.0, 1.0).var_axis(Axis(0), 0.0) / std.powi(2);
    //let var_perm = Array::range(0.0, eta as f64, 1.0).var_axis(Axis(0), 0.0) / snr_data;
    let std_perm = std; //f64::sqrt(var_perm.sum());

    //println!("{} {}",std_perm,std);
    // generate all the permutations
    let items = 0..eta;
    let mut all_perms = Vec::new();
    for perm in items.permutations(eta) {
        all_perms.push(perm);
    }
    let nperm = all_perms.len();

    // generate the shares and add noise
    let mut data = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));
    let mut perms = Array2::<u32>::random((n_traces, eta), Uniform::new(0, 2));

    let secrets = data.fold_axis(Axis(1), 0, |acc, y| (acc << 1) + y);

    // shuffling
    let mut tmp = Array1::<u32>::zeros(eta);
    let tmp = tmp.as_slice_mut().unwrap();
    for (mut x, mut perm) in data.genrows_mut().into_iter().zip(perms.genrows_mut()) {
        let x = x.as_slice_mut().unwrap();
        let perm = perm.as_slice_mut().unwrap();
        for i in 0..eta {
            tmp[i] = x[i];
        }
        let perm_x = &all_perms[rng.gen_range(0..nperm)];
        for i in 0..eta {
            x[i] = tmp[perm_x[i]];
            perm[i] = perm_x[i] as u32;
        }
    }
    // add noise to permutations and data
    let mut leakage = Array2::<f64>::random(data.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&data, |leakage, data| {
        *leakage = *data as f64 + std * *leakage
    });
    let mut leakage_perm = Array2::<f64>::random(perms.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage_perm.zip_mut_with(&perms, |leakage_perm, perms| {
        *leakage_perm = *perms as f64 + std_perm * *leakage_perm
    });

    // full entropy of the secret
    let h = 2usize.pow(eta as u32);

    // storage vectors
    let mut f_l_x = Array3::<f64>::zeros((n_traces, eta, 2)); // f_l_x
    let mut f_l_perm = Array3::<f64>::zeros((n_traces, eta, eta)); // f_l_x
    let mut f_l_mode = Array2::<f64>::zeros((n_traces, h));

    // compute gaussian distribution for each variable
    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }
    // compute gaussian distribution for each variable
    for (i, mut f_l_i) in f_l_perm.axis_iter_mut(Axis(2)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage_perm, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std_perm).powi(2)))
        });
    }

    // MI estimators
    let mut m = 0.0;
    let mut m2 = 0.0;
    let mut f_l_this_perm = Array1::<f64>::ones(n_traces);
    let mut f_l_this_data = Array1::<f64>::ones(n_traces);
    for perm in all_perms.iter() {
        // permutation
        f_l_this_perm.fill(1.0);
        for (c, f_l_c) in f_l_perm.axis_iter(Axis(1)).enumerate() {
            f_l_this_perm *= &f_l_c.slice(s![.., perm[c] as usize]);
        }

        // data
        for x in 0..h {
            f_l_this_data.fill(1.0);
            for (c, f_l_x) in f_l_x.axis_iter(Axis(1)).enumerate() {
                f_l_this_data *= &f_l_x.slice(s![.., (x >> (eta - 1 - perm[c])) & 0x1 as usize]);
            }
            let mut view = f_l_mode.slice_mut(s![.., x]);
            let tmp = &f_l_this_perm * &f_l_this_data;
            view.zip_mut_with(&tmp, |x, y| *x += y);
        }
    }

    for (f_l_s, secret) in f_l_mode.outer_iter().zip(secrets.iter()) {
        // Bayes' theorem
        let l = f64::log2(f_l_s[*secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));
        // update mi estimators
        m += l;
        m2 += l.powi(2);
    }
    (m, m2)
}

pub fn sample_mi_lin_m_shares(std: f64, d: usize, eta: usize, n_traces: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    // generate all the permutations
    let items = 0..eta;
    let mut all_perms = Vec::new();
    for perm in items.permutations(eta) {
        all_perms.push(perm);
    }
    let nperm = all_perms.len();

    // generate the shares and add noise
    let mut shares = Array3::<u32>::random((n_traces, d, eta), Uniform::new(0, 2));

    // reconstruct the secrets
    let bits = shares.fold_axis(Axis(1), 0, |acc, y| acc ^ y);
    let secrets = bits.fold_axis(Axis(1), 0, |acc, y| (acc << 1) + y);

    // shuffling
    let mut tmp = Array1::<u32>::zeros(eta);
    let tmp = tmp.as_slice_mut().unwrap();
    for mut x in shares.genrows_mut() {
        let x = x.as_slice_mut().unwrap();
        for i in 0..eta {
            tmp[i] = x[i];
        }
        let perm = &all_perms[rng.gen_range(0..nperm)];
        for i in 0..eta {
            x[i] = tmp[perm[i]];
        }
    }

    let mut leakage = Array3::<f64>::random(shares.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&shares, |leakage, shares| {
        *leakage = *shares as f64 + std * *leakage
    });

    // full entropy of the secret
    let h = 2usize.pow(eta as u32);

    // storage vectors
    let mut f_l_x = Array4::<f64>::zeros((n_traces, d, eta, 2)); // f_l_x
    let mut f_l_d = Array1::<f64>::ones(h); // store f(l|d) for a share (updated)
    let mut f_l_mode = Array1::<f64>::ones(h); // store f(l|d) for a share (updated)
    let mut f_l_s = Array1::<f64>::ones(h); // store f(l|x) for the complete secret

    // compute gaussian distribution for each variable
    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(3)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }

    // MI estimators
    let mut m = 0.0;
    let mut m2 = 0.0;

    let f_l_s = f_l_s.as_slice_mut().unwrap();
    let f_l_d = f_l_d.as_slice_mut().unwrap();
    let f_l_mode = f_l_mode.as_slice_mut().unwrap();

    // for each trace
    for (mut distri, secret) in f_l_x.axis_iter_mut(Axis(0)).zip(secrets.iter()) {
        for i in 0..h {
            f_l_s[i] = 1.0;
        }

        // for each share
        for mut distri in distri.axis_iter_mut(Axis(0)) {
            for i in 0..h {
                f_l_d[i] = 0.0;
            }
            for perm in all_perms.iter() {
                for i in 0..h {
                    f_l_mode[i] = 1.0;
                }
                // for each index
                for (s, mut distri) in distri.axis_iter_mut(Axis(0)).enumerate() {
                    let f_l_b = distri.as_slice_mut().unwrap();
                    for x in 0..h {
                        f_l_mode[x] *= f_l_b[(x >> (eta - 1 - perm[s])) & 0x1];
                    }
                }
                for i in 0..h {
                    f_l_d[i] += f_l_mode[i];
                }
            }
            // perform fwht
            fwht(f_l_d, h);
            for i in 0..h {
                f_l_s[i] *= f_l_d[i];
            }
        }
        fwht(f_l_s, h);

        // Bayes' theorem
        let l = f64::log2(f_l_s[*secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));

        // update mi estimators
        m += l;
        m2 += l.powi(2);
    }
    (m, m2)
}

pub fn sample_mi_lin_m_tuples(std: f64, d: usize, eta: usize, n_traces: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    // generate all the permutations
    let items = 0..eta;
    let mut all_perms = Vec::new();
    for perm in items.permutations(eta) {
        all_perms.push(perm);
    }
    let nperm = all_perms.len();

    let shares = Array3::<u32>::random((n_traces, eta, d), Uniform::new(0, 2));
    let mut leakage = Array3::<f64>::random(shares.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&shares, |leakage, shares| {
        *leakage = *shares as f64 + std * *leakage
    });

    let h = 2usize.pow(eta as u32);
    let mut f_l_x = Array4::<f64>::zeros((n_traces, eta, d, 2));
    let mut f_l_s = Array1::<f64>::ones(h);
    let mut f_l_mode = Array1::<f64>::ones(h);
    let mut f_l_b = Array2::<f64>::zeros((eta, 2));

    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(3)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }

    let bits = shares.fold_axis(Axis(2), 0, |acc, y| acc ^ y);
    let f_l_s = f_l_s.as_slice_mut().unwrap();
    let f_l_mode = f_l_mode.as_slice_mut().unwrap();

    let mut m = 0.0;
    let mut m2 = 0.0;
    for (mut distri, bits) in f_l_x.axis_iter_mut(Axis(0)).zip(bits.outer_iter()) {
        for i in 0..h {
            f_l_s[i] = 0.0;
        }
        for (mut distri, mut f_l_b) in distri
            .axis_iter_mut(Axis(0))
            .zip(f_l_b.axis_iter_mut(Axis(0)))
        {
            let f_l_b = f_l_b.as_slice_mut().unwrap();
            for i in 0..2 {
                f_l_b[i] = 1.0
            }
            for mut distri in distri.axis_iter_mut(Axis(0)) {
                let distri_slice = distri.as_slice_mut().unwrap();
                fwht(distri_slice, 2);
                for i in 0..2 {
                    f_l_b[i] *= distri_slice[i]
                }
            }
            fwht(f_l_b, 2);
        }
        // generate a random perm and the corresponding secret
        // we move the secret instead of the distributions to save
        // moves
        let perm = &all_perms[rng.gen_range(0..nperm)];
        let secret = bits
            .iter()
            .zip(perm.iter())
            .fold(0, |secret, (bits, perm)| secret | (bits << perm));

        // for each possible perm
        for i in 0..nperm {
            // reset the mode
            for i in 0..h {
                f_l_mode[i] = 1.0;
            }
            let perm = &all_perms[i];
            for (s, f_l_b) in f_l_b.outer_iter().enumerate() {
                // for each s
                for (x, f) in f_l_mode.iter_mut().enumerate() {
                    *f *= f_l_b[(x >> (eta - 1 - perm[s])) & 0x1];
                }
            }
            f_l_s
                .iter_mut()
                .zip(f_l_mode.iter())
                .for_each(|(f_l_s, f_l_mode)| *f_l_s += f_l_mode);
        }
        // Bayes' theorem
        let l = f64::log2(f_l_s[secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));

        // update mi estimators

        m += l;
        m2 += l.powi(2);
    }
    (m, m2)
}

pub fn sample_mi_lin_m(std: f64, d: usize, eta: usize, n_traces: usize) -> (f64, f64) {
    let shares = Array3::<u32>::random((n_traces, eta, d), Uniform::new(0, 2));
    let mut leakage = Array3::<f64>::random(shares.dim(), Normal::new(0.0, 1.0).unwrap());
    leakage.zip_mut_with(&shares, |leakage, shares| {
        *leakage = *shares as f64 + std * *leakage
    });

    let mut f_l_x = Array4::<f64>::zeros((n_traces, eta, d, 2));
    let mut f_l_s = Array1::<f64>::ones(2usize.pow(eta as u32));
    let mut f_l_b: [f64; 2] = [0.0; 2];
    for (i, mut f_l_i) in f_l_x.axis_iter_mut(Axis(3)).enumerate() {
        let i = i as f64;
        f_l_i.zip_mut_with(&leakage, |f, l| {
            *f = f64::exp(-0.5 * (((l - i) / std).powi(2)))
        });
    }

    let bits = shares.fold_axis(Axis(2), 0, |acc, y| acc ^ y);
    let secret = bits.fold_axis(Axis(1), 0, |acc, y| (acc << 1) + y);

    let f_l_s = f_l_s.as_slice_mut().unwrap();

    let mut m = 0.0;
    let mut m2 = 0.0;
    for (mut distri, secret) in f_l_x.axis_iter_mut(Axis(0)).zip(secret.iter()) {
        for i in 0..2usize.pow(eta as u32) {
            f_l_s[i] = 1.0;
        }
        for (s, mut distri) in distri.axis_iter_mut(Axis(0)).enumerate() {
            for i in 0..2 {
                f_l_b[i] = 1.0
            }
            for mut distri in distri.axis_iter_mut(Axis(0)) {
                let distri_slice = distri.as_slice_mut().unwrap();
                fwht(distri_slice, 2);
                for i in 0..2 {
                    f_l_b[i] *= distri_slice[i]
                }
            }
            let mut array = f_l_b;
            fwht(&mut array, 2);
            for (x, f) in f_l_s.iter_mut().enumerate() {
                *f *= array[(x >> (eta - 1 - s)) & 0x1];
            }
        }
        let l = f64::log2(f_l_s[*secret as usize] / f_l_s.iter().fold(0.0, |acc, x| acc + x));
        m += l;
        m2 += l.powi(2);
    }
    (m, m2)
}
pub fn sample_pr_from_modes(matrix: &ArrayView4<f64>, std: &ArrayView1<f64>) -> f64 {
    let mut rng = rand::thread_rng();

    let mut leakage = Array1::<f64>::random(std.dim(), Normal::new(0.0, 1.0).unwrap());
    let dims = matrix.shape();

    let secret = rng.gen_range(0..dims[0]);
    let pt = rng.gen_range(0..dims[1]);
    let mode = rng.gen_range(0..dims[2]);

    leakage
        .iter_mut()
        .zip(std.iter())
        .zip(matrix.slice(s![secret, pt, mode, ..]))
        .for_each(|((leakage, std), matrix)| *leakage = (*leakage * std) + matrix);

    let matrix_pt = matrix.slice(s![.., pt, .., ..]);
    let pr_x: Array1<f64> = matrix_pt
        .axis_iter(Axis(0))
        .map(|matrix_pt| {
            Zip::from(matrix_pt.genrows()).fold(0.0, |acc, row| {
                acc + f64::exp(
                    -0.5 * row
                        .iter()
                        .zip(leakage.iter())
                        .zip(std.iter())
                        .fold(0.0, |acc, ((u, l), std)| acc + ((u - l) / std).powi(2)),
                )
            })
        })
        .collect();
    f64::log2(pr_x[secret] / pr_x.sum())
}
