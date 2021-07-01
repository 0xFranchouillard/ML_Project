use ndarray::prelude::*;
use ndarray_rand::{rand};
use ndarray_rand::rand::Rng;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use ndarray_linalg::*;
use std::iter::FromIterator;
use osqp::{CscMatrix, Problem, Settings};
use libm::*;
use itertools::Itertools;

/// LINEAR MODEL ///
#[no_mangle]
pub extern "C" fn create_linear_model(mut input_dim: i32) -> *mut f32 {
    // let mut arr = Array::random((1, input_dim as usize), Uniform::new(-1., 1.));
    input_dim += 1;
    let mut arr = Vec::with_capacity(input_dim as usize);
    for _ in 0..input_dim {
        arr.push(rand::thread_rng()
            .gen_range(0.0..2.0) - 1.0);
    }

    let boxed_slice = arr.into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_regression_linear_model(model: *mut f32, dataset_inputs: *mut f32, dataset_expected_outputs: *mut f32, model_size: i32, dataset_inputs_size: i32) {
    let input_size = model_size as usize - 1;
    let sample_count = (dataset_inputs_size as usize) / input_size;
    let model = unsafe {
        from_raw_parts_mut(model, model_size as usize)
    };
    let dataset_inputs = unsafe {
        from_raw_parts(dataset_inputs, dataset_inputs_size as usize)
    };
    let dataset_expected_outputs = unsafe {
        from_raw_parts(dataset_expected_outputs, sample_count)
    };

    let all_inputs_array = Array::from_iter(dataset_inputs.iter().map(|f| f.clone()));
    let all_expected_outputs_array = Array::from_iter(dataset_expected_outputs.iter().map(|f| f.clone()));

    let all_inputs_matrix = all_inputs_array.into_shape((sample_count, input_size)).unwrap();

    let mut x = Array::<f32, _>::ones((sample_count, input_size + 1));
    x.slice_mut(s![..,1..]).assign(&all_inputs_matrix);
    let y = all_expected_outputs_array.into_shape((sample_count, 1 as usize)).unwrap();

    let xtx = x.t().dot(&x);
    let xtx_inv = xtx.inv().unwrap();
    let w = xtx_inv.dot(&x.t()).dot(&y);

    for i in 0..model_size as usize {
        model[i] = w[[i, 0]];
    }
}

#[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(model: *mut f32, dataset_inputs: *mut f32, dataset_expected_outputs: *mut f32, iterations_count: i32, alpha: f32, model_size: i32, dataset_inputs_size: i32) {
    let input_size = model_size as usize - 1;
    let sample_count = (dataset_inputs_size as usize) / input_size;
    let model = unsafe {
        from_raw_parts_mut(model, model_size as usize)
    };
    let dataset_inputs = unsafe {
        from_raw_parts(dataset_inputs, dataset_inputs_size as usize)
    };
    let dataset_expected_outputs = unsafe {
        from_raw_parts(dataset_expected_outputs, sample_count)
    };
    for _ in 0..iterations_count as usize {
        let k = rand::thread_rng().gen_range(0..sample_count);
        let xk = &dataset_inputs[(k * input_size) as usize..((k + 1) * input_size) as usize];
        let yk = dataset_expected_outputs[k * 1];
        let gxk = predict_linear_model_classification_slice(model, xk, model_size);

        model[0] += alpha * (yk - gxk) * 1.0;
        for i in 1..model_size as usize {
            model[i] += alpha * (yk - gxk) * xk[i - 1];
        }
    }
}

fn predict_linear_model_classification_slice(model: &[f32], inputs: &[f32], model_size: i32) -> f32 {
    let pred = predict_linear_model_regression_slice(model, inputs, model_size);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

fn predict_linear_model_regression_slice(model: &[f32], inputs: &[f32], model_size: i32) -> f32 {
    let mut sum_rslt = model[0];
    for i in 1..model_size as usize {
        sum_rslt += model[i] * inputs[i - 1];
    }
    return sum_rslt;
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let pred = predict_linear_model_regression(model, inputs, model_size);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let model = unsafe {
        from_raw_parts(model, model_size as usize)
    };
    let inputs = unsafe {
        from_raw_parts(inputs, model_size as usize - 1)
    };
    predict_linear_model_regression_slice(model, inputs, model_size)
    // let mut sum_rslt = model[0];
    // for i in 1..model_size as usize{
    //     sum_rslt += model[i] * inputs[i - 1];
    // }
    // return sum_rslt;
}

#[no_mangle]
pub extern "C" fn destroy_linear_model(model: *mut f32, model_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(model, model_size as usize, model_size as usize);
    }
}

/// MLP MODEL ///
#[derive(Debug)]
pub struct StructMLP {
    d: Vec<i32>,
    w: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    delta: Vec<Vec<f32>>,
}

impl StructMLP {
    pub extern "C" fn forward_pass(&mut self, sample_inputs: &Vec<f32>, is_classification: bool) {
        for j in 1..(self.d[0] + 1) as usize {
            self.x[0][j] = sample_inputs[j - 1];
        }
        for l in 1..(self.d.len()) as usize {
            for j in 1..(self.d[l] + 1) as usize {
                let mut sum_result = 0.0f32;
                for i in 0..(self.d[l - 1] + 1) as usize {
                    sum_result += self.w[l][i][j] * self.x[l - 1][i];
                }
                self.x[l][j] = sum_result;
                if (l < (self.d.len() - 1) as usize) || is_classification {
                    self.x[l][j] = self.x[l][j].tanh();
                }
            }
        }
    }
}

impl StructMLP {
    pub extern "C" fn train_stochastic_gradient_backpropagation(&mut self, flattened_data_inputs: &Vec<f32>, flattened_expected_outputs: &Vec<f32>, is_classification: bool, alpha: f32, iterations_count: i32) {
        let last = (self.d.len() - 1) as usize;
        let input_dim = self.d[0] as usize;
        let output_dim = self.d[last] as usize;
        let samples_count = flattened_data_inputs.len() / input_dim as usize;

        for _it in 0..iterations_count as usize {
            let k = rand::thread_rng().gen_range(0..samples_count) as usize;
            let sample_inputs = flattened_data_inputs[(k * input_dim)..((k + 1) * input_dim)].to_vec();
            let sample_expected_outputs = &flattened_expected_outputs[k * output_dim..(k + 1) * output_dim];

            self.forward_pass(&sample_inputs, is_classification);

            // Pour tous les neurones j de la dernière couche last on calcule delta[last][j]
            for j in 1..(self.d[last] + 1) as usize {
                self.delta[last][j] = self.x[last][j] - sample_expected_outputs[j - 1];
                if is_classification {
                    self.delta[last][j] = (1.0f32 - self.x[last][j] * self.x[last][j]) * self.delta[last][j]
                }
            }

            // On en déduit pour tous les autres neurones de l'avant dernière couche à la première
            for l in (1..last + 1).rev() {
                for i in 0..(self.d[l - 1] + 1) as usize {
                    let mut sum_result = 0.0f32;
                    for j in 1..(self.d[l] + 1) as usize {
                        sum_result += self.w[l][i][j] * self.delta[l][j];
                    }
                    self.delta[l - 1][i] = (1.0f32 - self.x[l - 1][i] * self.x[l - 1][i]) * sum_result;
                }
            }

            // Puis on met à jour tous les w[l][i][j]
            for l in 1..last + 1 {
                for i in 0..(self.d[l - 1] + 1) as usize {
                    for j in 1..(self.d[l] + 1) as usize {
                        self.w[l][i][j] += -alpha * self.x[l - 1][i] * self.delta[l][j]
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn create_mlp_model(npl: *mut i32, npl_size: i32) -> *mut StructMLP {
    let d = unsafe {
        from_raw_parts_mut(npl, npl_size as usize)
    };

    let mut w = Vec::with_capacity(npl_size as usize);

    for l in 0..npl_size as usize {
        if l == 0 {
            w.push(Vec::with_capacity(1 as usize));
            continue;
        }
        w.push(Vec::with_capacity((d[l - 1] + 1) as usize));

        for i in 0..(d[l - 1] + 1) as usize {
            w[l].push(Vec::with_capacity((d[l] + 1) as usize));

            for _k in 0..(d[l] + 1) as usize {
                w[l][i].push(rand::thread_rng()
                    .gen_range(0.0..2.0) - 1.0);
            }
        }
    }

    let mut x = Vec::with_capacity(npl_size as usize);
    for l in 0..npl_size as usize {
        x.push(Vec::with_capacity((d[l] + 1) as usize));

        for j in 0..(d[l] + 1) as usize {
            x[l].push(if j == 0 { 1.0 } else { 0.0 });
        }
    }

    let mut delta = Vec::with_capacity(npl_size as usize);
    for l in 0..npl_size as usize {
        delta.push(Vec::with_capacity((d[l] + 1) as usize));

        for _j in 0..(d[l] + 1) as usize {
            delta[l].push(0.0);
        }
    }

    let model = StructMLP {
        d: d.to_vec(),
        w,
        x,
        delta,
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

#[no_mangle]
pub extern "C" fn train_classification_stochastic_backprop_mlp_model(model: *mut StructMLP, flattened_data_inputs: *mut f32, flattened_data_inputs_size: i32, flattened_expected_outputs: *mut f32, flattened_expected_outputs_size: i32, alpha: f32, iterations_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let flattened_data_inputs = unsafe {
        from_raw_parts(flattened_data_inputs, flattened_data_inputs_size as usize)
    };
    let flattened_expected_outputs = unsafe {
        from_raw_parts(flattened_expected_outputs, flattened_expected_outputs_size as usize)
    };

    model.train_stochastic_gradient_backpropagation(&flattened_data_inputs.to_vec(), &flattened_expected_outputs.to_vec(), true, alpha, iterations_count)
}

#[no_mangle]
pub extern "C" fn train_regression_stochastic_backprop_mlp_model(model: *mut StructMLP, flattened_data_inputs: *mut f32, flattened_data_inputs_size: i32, flattened_expected_outputs: *mut f32, flattened_expected_outputs_size: i32, alpha: f32, iterations_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let flattened_data_inputs = unsafe {
        from_raw_parts(flattened_data_inputs, flattened_data_inputs_size as usize)
    };
    let flattened_expected_outputs = unsafe {
        from_raw_parts(flattened_expected_outputs, flattened_expected_outputs_size as usize)
    };

    model.train_stochastic_gradient_backpropagation(&flattened_data_inputs.to_vec(), &flattened_expected_outputs.to_vec(), false, alpha, iterations_count)
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_classification(model: *mut StructMLP, sample_inputs: *mut f32, sample_input_size: i32) -> *mut f32 {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let sample_inputs = unsafe {
        from_raw_parts(sample_inputs, sample_input_size as usize)
    };
    // dbg!(&model);
    // dbg!(&sample_inputs);

    model.forward_pass(&sample_inputs.to_vec(), true);

    let boxed_slice = model.x[model.d.len() - 1][1..].to_vec().into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    // dbg!(&arr_ref);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_regression(model: *mut StructMLP, sample_inputs: *mut f32, sample_input_size: i32) -> *mut f32 {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let sample_inputs = unsafe {
        from_raw_parts(sample_inputs, sample_input_size as usize)
    };

    model.forward_pass(&sample_inputs.to_vec(), false);

    let boxed_slice = model.x[model.d.len() - 1][1..].to_vec().into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn destroy_mlp_model(model: *mut StructMLP) {
    unsafe {
        let _ = Box::from_raw(model);
    }
}

/// SVM MODEL ///

pub fn quadratic_solver(big_matrix: Vec<Vec<f64>>, expected_outputs: &[f32], sample_count: usize) -> *mut f64 {
    let p = CscMatrix::from(&big_matrix).into_upper_tri();
    let q = Array::<f64, _>::ones((sample_count, 1)) * -1.0;
    let q = q.as_slice().unwrap();
    let mut a = Vec::with_capacity(sample_count + 1);
    for i in 0..sample_count + 1 {
        a.push(Vec::with_capacity(sample_count));
        for j in 0..sample_count {
            if i == 0 {
                a[i].push(expected_outputs[j] as f64);
            } else if i - 1 == j {
                a[i].push(1.0 as f64);
            } else {
                a[i].push(0.0 as f64);
            }
        }
    }
    let a = CscMatrix::from(&a);
    let l = Array::<f64, _>::zeros((sample_count + 1, 1));
    let l = l.as_slice().unwrap();
    let mut u = Array::<f64, _>::zeros((sample_count + 1, 1));
    for i in 1..sample_count + 1 {
        u[(i, 0)] = f64::MAX;
    }
    let u = u.as_slice().unwrap();

    let settings = Settings::default()
        .verbose(false);
    let mut prob = Problem::new(p, q, a, l, u, &settings).expect("failed to setup problem");
    let result = prob.solve();
    let alphas = result.x().unwrap();

    let boxed_slice = alphas.to_vec().into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn create_svm_model(sample_inputs_flat: *mut f64, expected_outputs: *mut f32, inputs_size: i32, sample_count: i32) -> *mut f32 {
    let inputs_size = inputs_size as usize;
    let sample_count = sample_count as usize;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, inputs_size * sample_count)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count)
    };

    let mut big_matrix = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        big_matrix.push(Vec::with_capacity(sample_count));
        let xi = Array::from((&sample_inputs_flat[(i * inputs_size)..((i + 1) * inputs_size)]).to_vec());
        for j in 0..sample_count {
            let xj = Array::from((&sample_inputs_flat[(j * inputs_size)..((j + 1) * inputs_size)]).to_vec());
            big_matrix[i].push(expected_outputs[i] as f64 * expected_outputs[j] as f64 * xi.t().dot(&xj));
        }
    }

    let alphas = unsafe {
        from_raw_parts(quadratic_solver(big_matrix, expected_outputs, sample_count), sample_count)
    };

    let mut tmp_w = Vec::with_capacity(inputs_size);
    for i in 0..inputs_size {
        tmp_w.push(0f32);
        for n in 0..sample_count {
            let xn = &sample_inputs_flat[(n * inputs_size)..((n + 1) * inputs_size)];
            let alpha_n = alphas[n] as f32;
            tmp_w[i] += alpha_n * expected_outputs[n] * xn[i] as f32;
        }
    }

    let sv_pos = alphas.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    // let mut sv_pos = sample_count-1;
    // for i in 0..sample_count {
    //     if result.x().unwrap()[i] > 0.001 {
    //         sv_pos = i;
    //         break;
    //     }
    // }

    let xsv_pos = &sample_inputs_flat[(sv_pos * inputs_size)..((sv_pos + 1) * inputs_size)];
    let mut sum_wx = 0.0;
    for n in 0..inputs_size {
        sum_wx += tmp_w[n] * xsv_pos[n] as f32;
    }
    let w0 = (1.0 / expected_outputs[sv_pos]) - sum_wx;

    let mut w = vec![w0];
    for i in 0..inputs_size {
        w.push(tmp_w[i]);
    }

    let boxed_slice = w.into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_svm(model: *mut f32, sample_inputs: *mut f32, sample_input_size: i32) -> f32 {
    let model = unsafe {
        Array::from(from_raw_parts(model, sample_input_size as usize).to_vec())
    };
    let sample_inputs = unsafe {
        Array::from(from_raw_parts(sample_inputs, sample_input_size as usize).to_vec())
    };
    model.dot(&sample_inputs)
}

#[derive(Debug)]
pub struct StructSVMKernelTrick {
    w0: f32,
    alphas: Vec<f64>,
    y: Vec<f32>,
    x: Vec<f32>,
    sample_count: usize,
    inputs_size: usize,
}

#[no_mangle]
pub extern "C" fn create_svm_kernel_trick_model(sample_inputs_flat: *mut f32, expected_outputs: *mut f32, inputs_size: i32, sample_count: i32) -> *mut StructSVMKernelTrick {
    let inputs_size = inputs_size as usize;
    let sample_count = sample_count as usize;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, inputs_size * sample_count)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count)
    };

    let mut big_matrix = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        big_matrix.push(Vec::with_capacity(sample_count));
        let xi = &sample_inputs_flat[(i * inputs_size)..((i + 1) * inputs_size)];
        for j in 0..sample_count {
            let xj = &sample_inputs_flat[(j * inputs_size)..((j + 1) * inputs_size)];
            big_matrix[i].push((expected_outputs[i] * expected_outputs[j] * radial_kernel(xi, xj)) as f64);
        }
    }

    let alphas = unsafe {
        from_raw_parts(quadratic_solver(big_matrix, expected_outputs, sample_count), sample_count)
    };

    let sv_pos = alphas.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let xsv_pos = &sample_inputs_flat[(sv_pos * inputs_size)..((sv_pos + 1) * inputs_size)];
    let mut sum = 0.0;
    for k in 0..sample_count {
        let xk = &sample_inputs_flat[(k * inputs_size)..((k + 1) * inputs_size)];
        let alpha_k = alphas[k] as f32;
        sum += alpha_k * expected_outputs[k] * radial_kernel(xk, xsv_pos);
    }
    let model = StructSVMKernelTrick {
        w0: (1.0 / expected_outputs[sv_pos]) - sum,
        alphas: alphas.to_vec(),
        y: expected_outputs.to_vec(),
        x: sample_inputs_flat.to_vec(),
        sample_count,
        inputs_size,
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

#[no_mangle]
pub extern "C" fn predict_svm_kernel_trick(model: *mut StructSVMKernelTrick, sample_inputs: *mut f32, sample_input_size: i32) -> f32 {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let sample_inputs = unsafe {
        from_raw_parts(sample_inputs, sample_input_size as usize)
    };
    let mut result_sum = 0.0;
    for k in 0..model.sample_count {
        let xk = &model.x[(k * model.inputs_size)..((k + 1) * model.inputs_size)];
        if k == 0 {
            result_sum = (model.alphas[k] as f32) * model.y[k] * radial_kernel(xk, sample_inputs);
        } else {
            result_sum += (model.alphas[k] as f32) * model.y[k] * radial_kernel(xk, sample_inputs);
        }
    }
    result_sum + model.w0
}

pub fn radial_kernel(xn: &[f32], xm: &[f32]) -> f32 {
    expf((Array::from(xn.to_vec()).dot(&Array::from(xn.to_vec())) * -1.0) as f32)
        * expf((Array::from(xm.to_vec()).dot(&Array::from(xm.to_vec())) * -1.0) as f32)
        * expf(((Array::from(xn.to_vec()) * 2.0).dot(&Array::from(xm.to_vec()))) as f32)
}

/// RBF MODEL ///
#[derive(Debug)]
pub struct StructRBFKCenter {
    w: Vec<f32>,
    x: Vec<Vec<f32>>,
    gamma: f32,
}

#[no_mangle]
pub extern "C" fn create_rbf_k_center_model(input_dim: i32, cluster_num: i32, gamma: f32) -> *mut StructRBFKCenter {
    let mut w = Vec::with_capacity(cluster_num as usize);
    for _ in 0..cluster_num as usize {
        w.push(0f32);
    }
    let mut x = Vec::with_capacity(cluster_num as usize);
    for i in 0..cluster_num as usize {
        x.push(Vec::with_capacity(input_dim as usize));
        for _ in 0..input_dim as usize {
            x[i].push(0f32);
        }
    }
    let model = StructRBFKCenter {
        w,
        x,
        gamma,
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

pub fn euclid(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0f32;
    for i in 0..(x.len()) {
        result += powf(y[i] - x[i], 2f32);
    }
    sqrtf(result)
}

pub fn get_rand_sites(data: &[f32], cluster_num: i32, sample_count: i32, inputs_size: i32) -> Vec<Vec<f32>> {
    let mut sites = Vec::with_capacity(cluster_num as usize);
    let mut data_copy_size = sample_count;
    let mut data_copy = data.to_vec();
    for _ in 0..cluster_num {
        let initial_center = rand::thread_rng().gen_range(0..data_copy_size);
        let data_initial_center = &data_copy[(initial_center * inputs_size) as usize..((initial_center + 1) * inputs_size) as usize];
        sites.push(data_initial_center.to_vec());
        data_copy_size -= 1;
        data_copy.remove(initial_center as usize);
    }
    sites
}

pub fn mean(cluster: &[&[f32]], inputs_size: i32) -> Vec<f32> {
    let mut average = Vec::with_capacity(inputs_size as usize);
    for dimension in 0..inputs_size as usize {
        average.push(0f32);
        for points in 0..cluster.len() {
            average[dimension] += cluster[points][dimension];
        }
        average[dimension] /= cluster.len() as f32;
    }
    average
}

pub fn lloyd(data: &[f32], cluster_num: i32, iterations: i32, sample_count: i32, inputs_size: i32) -> Vec<f32> {
    if cluster_num == sample_count {
        return data.to_vec();
    }
    let mut clusters = Vec::with_capacity(cluster_num as usize);
    for _ in 0..cluster_num {
        clusters.push(Vec::new());
    }
    let mut sites = get_rand_sites(data, cluster_num, sample_count, inputs_size).to_vec();
    for _ in 0..iterations as usize {
        for point in 0..sample_count as usize {
            let data_point = &data[(point * inputs_size as usize)..((point + 1) * inputs_size as usize)];
            let mut closest_site_number = 0usize;
            let mut closest_site_distance = euclid(sites[closest_site_number].as_slice(), data_point);
            for site_number in 1..sites.len() {
                let site_distance = euclid(sites[site_number].as_slice(), data_point);
                if site_distance < closest_site_distance {
                    closest_site_number = site_number;
                    closest_site_distance = site_distance;
                }
            }
            clusters[closest_site_number].push(data_point);
        }
        for m in 0..cluster_num as usize {
            sites[m] = mean(clusters[m].as_slice(), inputs_size);
            clusters[m].clear();
        }
    }
    let mut sites_flat = Vec::with_capacity((cluster_num * inputs_size) as usize);
    for i in 0..cluster_num as usize {
        for j in 0..inputs_size as usize {
            sites_flat.push(sites[i][j]);
        }
    }
    sites_flat
}

#[no_mangle]
pub extern "C" fn train_regression_rbf_k_center_model(model: *mut StructRBFKCenter, sample_inputs_flat: *mut f32, expected_outputs: *mut f32, inputs_size: i32, sample_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let cluster_num = model.w.len() as i32;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, (inputs_size * sample_count) as usize)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count as usize)
    };
    let cluster_points = lloyd(sample_inputs_flat, cluster_num, 10, sample_count, inputs_size);

    let mut phi = Array::default((sample_count as usize, cluster_num as usize));
    for i in 0..sample_count as usize {
        let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
        for j in 0..cluster_num as usize {
            let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1) * inputs_size as usize)];
            phi[(i, j)] = expf(-model.gamma * euclid(xi, cluster_pointsj) * euclid(xi, cluster_pointsj));
            for n in 0..inputs_size as usize {
                model.x[j][n] = cluster_pointsj[n];
            }
        }
    }

    let y = Array::from(expected_outputs.to_vec());
    let phitphi = phi.t().dot(&phi);
    let phitphi_inv = phitphi.inv().unwrap();
    let w = (phitphi_inv.dot(&phi.t())).dot(&y);

    for i in 0..cluster_num as usize {
        model.w[i] = w[i];
    }
}

#[no_mangle]
pub extern "C" fn predict_rbf_k_center_model_regression(model: *mut StructRBFKCenter, inputs: *mut f32) -> f32 {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let inputs = unsafe {
        from_raw_parts(inputs, model.x[0].len())
    };
    predict_rbf_k_center_model_regression_slice(model, inputs)
    // let mut result = 0f32;
    // for i in 0..model.w.len() {
    //     result += model.w[i] * expf(-model.gamma * euclid(inputs, model.x[i].as_slice()) * euclid(inputs, model.x[i].as_slice()));
    // }
    // result
}

#[no_mangle]
pub extern "C" fn train_rosenblatt_rbf_k_center_model(model: *mut StructRBFKCenter, sample_inputs_flat: *mut f32, expected_outputs: *mut f32, iterations_count: i32, alpha: f32, inputs_size: i32, sample_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let cluster_num = model.w.len() as i32;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, (inputs_size * sample_count) as usize)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count as usize)
    };
    let cluster_points = lloyd(sample_inputs_flat, cluster_num, 10, sample_count, inputs_size);

    let mut phi = Vec::with_capacity(sample_count as usize);
    for i in 0..sample_count as usize {
        phi.push(Vec::with_capacity(cluster_num as usize));
        let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
        for j in 0..cluster_num as usize {
            let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1) * inputs_size as usize)];
            phi[i].push(expf(-model.gamma * euclid(xi, cluster_pointsj) * euclid(xi, cluster_pointsj)));
            for n in 0..inputs_size as usize {
                model.x[j][n] = cluster_pointsj[n];
            }
        }
    }

    // let mut phi = Array::default((sample_count as usize, cluster_num as usize));
    // for i in 0..sample_count as usize {
    //     let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
    //     for j in 0..cluster_num as usize {
    //         let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1) * inputs_size as usize)];
    //         phi[(i, j)] = expf(-model.gamma * euclid(xi, cluster_pointsj) * euclid(xi, cluster_pointsj));
    //         for n in 0..inputs_size as usize {
    //             model.x[j][n] = cluster_pointsj[n];
    //         }
    //     }
    // }
    //
    // let y = Array::from(expected_outputs.to_vec());
    // let phitphi = phi.t().dot(&phi);
    // let phitphi_inv = phitphi.inv().unwrap();
    // let w = (phitphi_inv.dot(&phi.t())).dot(&y);
    //
    // for i in 0..cluster_num as usize {
    //     model.w[i] = w[i];
    // }

    for _ in 0..iterations_count as usize {
        // let k = rand::thread_rng().gen_range(0..cluster_num) as usize;
        let k = rand::thread_rng().gen_range(0..sample_count) as usize;
        // let xk = &cluster_points[(k * inputs_size as usize)..((k + 1) * inputs_size as usize)];
        let x = &sample_inputs_flat[(k * inputs_size as usize)..((k + 1) * inputs_size as usize)];
        let xk = phi[k].clone();
        let yk = expected_outputs[k * 1];
        let gx = predict_rbf_k_center_model_classification_slice(model, x);
        // let gxk = predict_rbf_k_center_model_classification_slice(model, xk.as_slice());

        for i in 0..cluster_num as usize {
            model.w[i] += alpha * (yk - gx) * xk[i];
        }
    }
}

fn predict_rbf_k_center_model_classification_slice(model: &StructRBFKCenter, inputs: &[f32]) -> f32 {
    let pred = predict_rbf_k_center_model_regression_slice(model, inputs);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

fn predict_rbf_k_center_model_regression_slice(model: &StructRBFKCenter, inputs: &[f32]) -> f32 {
    let mut result = 0f32;
    for i in 0..model.w.len() {
        result += model.w[i] * expf(-model.gamma * euclid(inputs, model.x[i].as_slice()) * euclid(inputs, model.x[i].as_slice()));
    }
    result
}

#[no_mangle]
pub extern "C" fn predict_rbf_k_center_model_classification(model: *mut StructRBFKCenter, inputs: *mut f32) -> f32 {
    let pred = predict_rbf_k_center_model_regression(model, inputs);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

#[no_mangle]
pub extern "C" fn destroy_rbf_k_center_model(model: *mut StructRBFKCenter) {
    unsafe {
        let _ = Box::from_raw(model);
    }
}