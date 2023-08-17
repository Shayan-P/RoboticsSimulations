#include "casadi/casadi.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/algorithm/aba.hpp>

#include "utils.h"

using namespace casadi;
using namespace pinocchio;
using namespace std;



int main() {
  Model _model;
  auto path = "../ANYmal.urdf";
  pinocchio::urdf::buildModel(path, _model);
  auto model = _model.cast<Scalar>();
  cout << "model: " << model.name << endl;

  pinocchio::DataTpl<Scalar> data(model);
 
  auto q = SX::sym("q", model.nq);
  auto dq = SX::sym("dq", model.nv);
  auto ddq = SX::sym("ddq", model.nv);
  auto tau = SX::sym("tau", model.nv);

  rnea(model, data, cas_to_eig(q), cas_to_eig(dq), cas_to_eig(ddq));

  tau = eig_to_cas(data.tau);

  Function f("inverse_dynamics", {q, dq, ddq}, {tau}, {"q", "dq", "ddq"}, {"tau"});
  f.save("inverse_dynamics.func");

  cout << f.serialize() << endl;

  // Eigen::VectorXd q = pinocchio::neutral(model);
  // Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
  // Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);
 
  // const Eigen::VectorXd & tau = pinocchio::rnea(model,data,q,v,a);
  // std::cout << "tau = " << tau.transpose() << std::endl;
}
