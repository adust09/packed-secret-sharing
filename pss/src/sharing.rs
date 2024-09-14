use crate::circuit::WireID; //  SNGに依存する
use crate::math::galois::GF;
use crate::math::lagrange_coeffs;
use crate::ProtoErrorKind;
use ndarray::{s, Array2, ArrayView, ArrayView1, ArrayView2};
use rand::Rng;

pub type PackedShare<const W: u8> = GF<W>;
