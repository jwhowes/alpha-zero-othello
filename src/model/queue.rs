use std::{iter::zip, sync::mpsc};

use candle_core::{Result, Tensor};

use crate::model::vit::ViT;

pub type EvaluateResponse = (Vec<Vec<f32>>, f32);

pub type EvaluateRequest = (Tensor, oneshot::Sender<EvaluateResponse>);

pub fn evaluation_thread(model: &ViT, queue_rx: mpsc::Receiver<EvaluateRequest>) -> Result<()> {
    loop {
        let (board, response_tx) = match queue_rx.recv() {
            Ok(x) => x,
            Err(_) => break,
        };

        let mut boards = vec![board];
        let mut response_txs = vec![response_tx];

        while let Ok((board, response_tx)) = queue_rx.try_recv() {
            boards.push(board);
            response_txs.push(response_tx);
        }

        let batch = Tensor::stack(boards.as_slice(), 0)?;

        let (prior, value) = model.forward(&batch)?;

        let priors = prior.to_vec3::<f32>()?;
        let values = value.to_vec1::<f32>()?;

        for ((prior, value), tx) in zip(zip(priors, values), response_txs) {
            tx.send((prior, value)).unwrap();
        }
    }

    Ok(())
}
