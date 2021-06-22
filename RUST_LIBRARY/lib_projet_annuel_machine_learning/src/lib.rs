use ndarray::prelude::*;
use ndarray_rand::{rand};
use ndarray_rand::rand::Rng;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use ndarray_linalg::*;
use std::iter::FromIterator;
use osqp::{CscMatrix, Problem, Settings};

#[no_mangle]
pub extern "C" fn create_linear_model(mut input_dim: i32) -> *mut f32{
    // let mut arr = Array::random((1, input_dim as usize), Uniform::new(-1., 1.));
    input_dim += 1;
    let mut arr = Vec::with_capacity(input_dim as usize);
    for _ in 0..input_dim{
        arr.push(rand::thread_rng()
            .gen_range(0.0..2.0)-1.0);
    }

    let boxed_slice = arr.into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_regression_linear_model(model: *mut f32, dataset_inputs: *mut f32, dataset_expected_outputs: *mut f32, model_size: i32, dataset_inputs_size: i32) {
    let input_size = model_size as usize -1;
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

    let all_inputs_array = Array::from_iter(dataset_inputs.iter().map(|f|f.clone()));
    let all_expected_outputs_array = Array::from_iter(dataset_expected_outputs.iter().map(|f|f.clone()));

    let all_inputs_matrix = all_inputs_array.into_shape((sample_count,input_size)).unwrap();

    let mut x = Array::<f32, _>::ones((sample_count,input_size+1));
    x.slice_mut(s![..,1..]).assign(&all_inputs_matrix);
    let y = all_expected_outputs_array.into_shape((sample_count,1 as usize)).unwrap();

    let xtx = x.t().dot(&x);
    let xtx_inv = xtx.inv().unwrap();
    let w = xtx_inv.dot(&x.t()).dot(&y);

    for i in 0..model_size as usize {
        model[i] = w[[i,0]];
    }
}

    #[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(model: *mut f32, dataset_inputs: *mut f32, dataset_expected_outputs: *mut f32, iterations_count: i32, alpha: f32, model_size: i32, dataset_inputs_size: i32) {
    let input_size = model_size as usize -1;
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
        let gxk = predict_linear_model_classification_slice(model,xk ,model_size);

        model[0] += alpha * (yk - gxk) * 1.0;
        for i in 1..model_size as usize {
            model[i] += alpha * (yk - gxk) * xk[i - 1];
        }
    }
}

#[no_mangle]
fn predict_linear_model_classification_slice(model: &[f32], inputs: &[f32], model_size: i32) -> f32 {
    let pred = predict_linear_model_regression_slice(model,inputs,model_size);
    return if pred >= 0.0 {1.0} else {-1.0}
}

#[no_mangle]
fn predict_linear_model_regression_slice(model: &[f32], inputs: &[f32], model_size: i32) -> f32 {
    let mut sum_rslt = model[0];
    for i in 1..model_size as usize{
        sum_rslt += model[i] * inputs[i - 1];
    }
    return sum_rslt;
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let pred = predict_linear_model_regression(model,inputs,model_size);
    return if pred >= 0.0 {1.0} else {-1.0}
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let model = unsafe {
        from_raw_parts(model,model_size as usize)
    };
    let inputs = unsafe {
        from_raw_parts(inputs,model_size as usize -1)
    };
    let mut sum_rslt = model[0];
    for i in 1..model_size {
        sum_rslt += model[i as usize] * inputs[i as usize -1];
    }
    return sum_rslt;
}

#[no_mangle]
pub extern "C" fn destroy_linear_model(model: *mut f32, model_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(model, model_size as usize, model_size as usize);
    }
}

#[derive(Debug)]
pub struct StructMLP {
    d: Vec<i32>,
    w: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    delta: Vec<Vec<f32>>
}

impl StructMLP {
    pub extern "C" fn forward_pass(&mut self, sample_inputs: &Vec<f32>, is_classification: bool){
        for j in 1..(self.d[0] + 1) as usize {
            self.x[0][j] = sample_inputs[j -1];
        }
        for l in 1..(self.d.len()) as usize {
            for j in 1..(self.d[l] + 1) as usize{
                let mut sum_result = 0.0f32;
                for i in 0..(self.d[l -1] + 1) as usize{
                    sum_result += self.w[l][i][j] * self.x[l-1][i];
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
        let samples_count = flattened_data_inputs.len()/input_dim as usize;

        for _it in 0..iterations_count as usize {
            let k = rand::thread_rng().gen_range(0..samples_count) as usize;
            let sample_inputs = flattened_data_inputs[(k * input_dim)..((k+1) * input_dim)].to_vec();
            let sample_expected_outputs = &flattened_expected_outputs[k * output_dim..(k+1) * output_dim];

            self.forward_pass(&sample_inputs, is_classification);

            // Pour tous les neurones j de la dernière couche last on calcule delta[last][j]
            for j in 1..(self.d[last] + 1) as usize {
                self.delta[last][j] = self.x[last][j] - sample_expected_outputs[j-1];
                if is_classification {
                    self.delta[last][j] = (1.0f32 - self.x[last][j] * self.x[last][j]) * self.delta[last][j]
                }
            }

            // On en déduit pour tous les autres neurones de l'avant dernière couche à la première
            for l in (1..last+1).rev() {
                for i in 0..(self.d[l - 1] + 1) as usize {
                    let mut sum_result = 0.0f32;
                    for j in 1..(self.d[l] + 1) as usize {
                        sum_result += self.w[l][i][j] * self.delta[l][j];
                    }
                    self.delta[l-1][i] = (1.0f32 - self.x[l-1][i] * self.x[l-1][i]) * sum_result;
                }
            }

            // Puis on met à jour tous les w[l][i][j]
            for l in 1..last+1 {
                for i in 0..(self.d[l-1] + 1) as usize {
                    for j in 1..(self.d[l] + 1) as usize {
                        self.w[l][i][j] += - alpha * self.x[l - 1][i] * self.delta[l][j]
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn create_mlp_model(npl: *mut i32,  npl_size: i32) -> *mut StructMLP{
    let d = unsafe {
        from_raw_parts_mut(npl, npl_size as usize)
    };

    let mut w= Vec::with_capacity(npl_size as usize);

    for l in  0..npl_size as usize {
        if l == 0 {
            w.push(Vec::with_capacity(1 as usize));
            continue;
        }
        w.push(Vec::with_capacity((d[l -1]+1) as usize));

        for i in 0..(d[l -1] + 1) as usize {
            w[l].push(Vec::with_capacity((d[l]+1) as usize));

            for _k in 0..(d[l] + 1) as usize {
                w[l][i].push(rand::thread_rng()
                    .gen_range(0.0..2.0)-1.0);
            }
        }
    }

    let mut x= Vec::with_capacity(npl_size as usize);
    for l in 0..npl_size as usize {
        x.push(Vec::with_capacity((d[l] + 1) as usize));

        for j in 0..(d[l] + 1) as usize{
            x[l].push(if j == 0 {1.0} else {0.0});
        }
    }

    let mut delta = Vec::with_capacity(npl_size as usize);
    for l in 0..npl_size as usize {
        delta.push(Vec::with_capacity((d[l] + 1) as usize));

        for _j in 0..(d[l] + 1) as usize {
            delta[l].push(0.0);
        }
    }

    let model = StructMLP{
        d: d.to_vec(),
        w,
        x,
        delta
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

#[no_mangle]
pub extern "C" fn train_classification_stochastic_backprop_mlp_model(model: *mut StructMLP, flattened_data_inputs: *mut f32, flattened_data_inputs_size: i32, flattened_expected_outputs: *mut f32, flattened_expected_outputs_size: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let flattened_data_inputs = unsafe{
        from_raw_parts(flattened_data_inputs,flattened_data_inputs_size as usize)
    };
    let flattened_expected_outputs = unsafe{
        from_raw_parts(flattened_expected_outputs,flattened_expected_outputs_size as usize)
    };

    model.train_stochastic_gradient_backpropagation(&flattened_data_inputs.to_vec(),&flattened_expected_outputs.to_vec(),true, 0.01f32, 10000)
}

#[no_mangle]
pub extern "C" fn train_regression_stochastic_backprop_mlp_model(model: *mut StructMLP, flattened_data_inputs: *mut f32, flattened_data_inputs_size: i32, flattened_expected_outputs: *mut f32, flattened_expected_outputs_size: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let flattened_data_inputs = unsafe{
        from_raw_parts(flattened_data_inputs,flattened_data_inputs_size as usize)
    };
    let flattened_expected_outputs = unsafe{
        from_raw_parts(flattened_expected_outputs,flattened_expected_outputs_size as usize)
    };

    model.train_stochastic_gradient_backpropagation(&flattened_data_inputs.to_vec(),&flattened_expected_outputs.to_vec(),false, 0.01f32, 10000)
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_classification(model: *mut StructMLP, sample_inputs: *mut f32, sample_input_size: i32) -> *mut f32{
    let model = unsafe {
      model.as_mut().unwrap()
    };
    let sample_inputs = unsafe {
        from_raw_parts(sample_inputs, sample_input_size as usize)
    };
    // dbg!(&model);
    dbg!(&sample_inputs);

    model.forward_pass(&sample_inputs.to_vec(), true);

    let boxed_slice = model.x[model.d.len()-1][1..].to_vec().into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    dbg!(&arr_ref);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_regression(model: *mut StructMLP, sample_inputs: *mut f32, sample_input_size: i32) -> *mut f32{
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let sample_inputs = unsafe {
        from_raw_parts(sample_inputs, sample_input_size as usize)
    };

    model.forward_pass(&sample_inputs.to_vec(), false);

    let boxed_slice = model.x[model.d.len()-1][1..].to_vec().into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn destroy_mlp_model(model: *mut StructMLP){
    unsafe {
        let _ = Box::from_raw(model);
    }
}

#[no_mangle]
pub extern "C" fn train_svm(sample_inputs_flat: *mut f32, expected_outputs: *mut f32, inputs_size: i32, sample_count: i32) {
    let inputs_size = inputs_size as usize;
    let sample_count = sample_count as usize;
    let sample_inputs_flat = unsafe {
        from_raw_parts_mut(sample_inputs_flat, inputs_size*sample_count)
    };
    let expected_outputs = unsafe {
        from_raw_parts_mut(expected_outputs, sample_count)
    };

    let mut big_matrix = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        big_matrix.push(Vec::with_capacity(sample_count));
        let xi = Array::from((&sample_inputs_flat[(i * inputs_size)..((i + 1) * inputs_size)]).to_vec());
        for j in 0..sample_count {
            let xj = Array::from((&sample_inputs_flat[(j * inputs_size)..((j + 1) * inputs_size)]).to_vec());
            big_matrix[i].push((expected_outputs[i] * expected_outputs[j] * xi.t().dot(&xj)) as f64);
        }
    }
    let P = CscMatrix::from(&big_matrix).into_upper_tri();
    let q = (Array::ones((sample_count,1)) * -1.0).as_slice().unwrap();
    let A = CscMatrix::from((Array::<f64,_>::from(expected_outputs.to_vec()).reversed_axes()).as_slice());
    let l = Array::zeros((sample_count,1)).as_slice().unwrap();
    let u = Array::zeros((sample_count,1)).as_slice().unwrap();
    dbg!(&A);
    dbg!(&q);
    let settings = Settings::default().verbose(true);
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");
    let result = prob.solve();
    println!("{:?}", result.x().expect("failed to solve problem"));

}