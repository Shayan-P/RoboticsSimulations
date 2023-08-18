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
  pinocchio::JointModelFreeFlyer root_joint;

  auto path = "../ANYmal.urdf";
  pinocchio::urdf::buildModel(path, root_joint, _model);
  auto model = _model.cast<Scalar>();
  cout << "model: " << model.name << endl;
  cout << "nq: " << model.nq << endl << "nv: " << model.nv << endl << "njoints: " << model.njoints << endl;

  for(int i = 0; i < model.njoints; i++) {
    cout << model.joints[i] << endl;
  }
  pinocchio::DataTpl<Scalar> data(model);
 
  auto q = SX::sym("q", model.nq);
  auto dq = SX::sym("dq", model.nv);
  auto ddq = SX::sym("ddq", model.nv);
  auto tau = SX::sym("tau", model.nv);

  rnea(model, data, cas_to_eig(q), cas_to_eig(dq), cas_to_eig(ddq));
  tau = eig_to_cas(data.tau);

  auto mass_matrix = eig_to_cas(crba(model, data, cas_to_eig(q)));

  Function dynamics("dynamics", {q, dq, ddq}, {tau}, {"q", "dq", "ddq"}, {"tau"});
  dynamics.save("../casadi_functions/dynamics.func");

  Function M("M", {q}, {mass_matrix}, {"q"}, {"M"});
  M.save("../casadi_functions/M.func");
}
