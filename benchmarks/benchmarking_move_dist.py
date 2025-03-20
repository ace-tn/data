import torch
import torch.distributed as dist
from acetn.ipeps import Ipeps
from acetn.renormalization.directional_mover import DirectionalMover
from acetn.utils.benchmarking import record_runtime_ave
import sys

@record_runtime_ave
def mock_up_down_move(mover, ipeps, y1=0, y2=0):
    rank, ws = ipeps.rank, ipeps.world_size
    proj1u, proj2u = {}, {}
    proj1d, proj2d = {}, {}
    bD = ipeps.dims["bond"]
    cD = ipeps.dims["chi"]
    for x_shift in range(0, 2*ipeps.nx, ws):
        proj1u_i, proj2u_i = torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device), torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device)
        proj1d_i, proj2d_i = torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device), torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device)
        if rank < ws//2:
            xi = rank + x_shift//ws
            if xi < ipeps.nx:
                proj1u_i, proj2u_i = mover.calculate_up_projectors(ipeps, xi, y1)
        elif rank >= ws//2:
            xi = rank - ws//2 + x_shift//ws
            if xi < ipeps.nx:
                proj1d_i, proj2d_i = mover.calculate_down_projectors(ipeps, xi, y2)

        proj1u_all, proj2u_all = [torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device),]*ws, [torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device),]*ws
        proj1d_all, proj2d_all = [torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device),]*ws, [torch.empty(cD,bD,bD,cD, dtype=ipeps.dtype, device=ipeps.device),]*ws

        torch.cuda.nvtx.range_push("all_gather [color=#0000FF]")
        dist.all_gather(proj1u_all, proj1u_i)
        dist.all_gather(proj2u_all, proj2u_i)
        dist.all_gather(proj1d_all, proj1d_i)
        dist.all_gather(proj2d_all, proj2d_i)
        torch.cuda.nvtx.range_pop()

        for xi in range(ws//2):
            proj1u[xi + x_shift//ws] = proj1u_all[xi].to(rank)
            proj2u[xi + x_shift//ws] = proj2u_all[xi].to(rank)
            proj1d[xi + x_shift//ws] = proj1d_all[xi + ws//2].to(rank)
            proj2d[xi + x_shift//ws] = proj2d_all[xi + ws//2].to(rank)

    for xi in range(ipeps.nx):
        xj = (xi+1) % ipeps.nx
        yj = (y1-1+ipeps.ny) % ipeps.ny
        s1 = (xi,y1)
        s2 = (xi,yj)
        mover.renormalize_boundary(ipeps, proj1u, proj2u, s1, s2, xi, xj, k=1)

    for xi in range(ipeps.nx):
        xj = (xi-1+ipeps.nx) % ipeps.nx
        yj = (y2+1) % ipeps.ny
        s1 = (xi,y2)
        s2 = (xi,yj)
        mover.renormalize_boundary(ipeps, proj1d, proj2d, s1, s2, xi, xj, k=3)


if __name__=="__main__":
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 3
    dims['chi'] = dims['bond']**2

    if len(sys.argv) == 2:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = dims['bond']**2
    elif len(sys.argv) == 3:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = int(sys.argv[2])

    dtype = torch.float64
    device = torch.device('cuda')

    projector_type = 'full-system'
    ctmrg_steps = 0

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN':{
            'nx': 2,
            'ny': 2,
            'dims': dims
        },
        'model':{
            'name': 'heisenberg',
            'J': 1.0,
        },
        'ctmrg':{
            'steps': ctmrg_steps,
            'projectors': projector_type,
        },
    }

    ipeps = Ipeps(ipeps_config)
    mover = DirectionalMover(ipeps.config.ctmrg)

    print("start benchmarking up_down_move...")
    mock_move_runtime_ave = mock_up_down_move(mover, ipeps)
    print("done benchmarking")
    print(f"Average runtime: {mock_move_runtime_ave:.7f} s")
