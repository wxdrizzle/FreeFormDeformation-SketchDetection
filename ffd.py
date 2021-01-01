from PIL import Image
from pylab import *
import torch
from matplotlib import pyplot as plt


def torch_Bspline(uv, kl):
    return (
        torch.where(kl == 0, (1 - uv) ** 3 / 6,
                    torch.where(kl == 1, uv ** 3 / 2 - uv ** 2 + 2 / 3,
                                torch.where(kl == 2, (-3 * uv ** 3 + 3 * uv ** 2 + 3 * uv + 1) / 6,
                                            torch.where(kl == 3, uv ** 3 / 6, torch.zeros_like(uv)))))
    )


def torch_transformation(pos, mesh, delta):
    pos_reg = pos / delta
    pos_floor = pos_reg.floor().long()
    uv = pos_reg - pos_floor
    ij = pos_floor - 1
    kls = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4))).flatten(1)
    result = torch.zeros_like(pos).float()
    for kl in kls.T:
        B = torch_Bspline(uv, kl.view(2, 1, 1))
        pivots = (ij + 1 + kl.view(2, 1, 1)).clamp(0, mesh.size(-1) - 1)
        result += B.prod(0, keepdim=True) * mesh[:, pivots[0], pivots[1]]
    return result


def torch_interpolation(pos, img):
    pos_floor = pos.floor().long()
    uv = pos - pos_floor
    ij = pos_floor
    get_img = lambda b, a: img[a.clamp(0, img.size(0) - 1), b.clamp(0, img.size(0) - 1)]
    return (
            ((1 - uv[0]) * (1 - uv[1])).unsqueeze(-1) * get_img(ij[0], ij[1]) +
            ((1 - uv[0]) * uv[1]).unsqueeze(-1) * get_img(ij[0], ij[1] + 1) +
            (uv[0] * (1 - uv[1])).unsqueeze(-1) * get_img(ij[0] + 1, ij[1]) +
            (uv[0] * uv[1]).unsqueeze(-1) * get_img(ij[0] + 1, ij[1] + 1)
    )


def reverse_mapping(img, mesh, delta):
    mesh = torch.transpose(mesh,1,2)
    pixel_grid = torch.stack(torch.meshgrid(torch.arange(img.shape[0]), torch.arange(img.shape[0])))
    return torch_interpolation(torch_transformation(pixel_grid, mesh, delta), img)


def compute_warped_img(mesh_trans, img, delta, iter_num=5, lr=40):
    img = torch.tensor(img)
    mesh_trans = torch.tensor(mesh_trans)
    result = (128 * torch.ones_like(img)).float().requires_grad_(True)
    opt = torch.optim.Adam([result], lr=lr)
    for iter in range(iter_num):
        img_cycled = reverse_mapping(result.clamp(0, 255), mesh_trans, delta)
        ssd = ((img_cycled - img) ** 2).mean()
        opt.zero_grad()
        ssd.backward()
        opt.step()
    result = np.clip(result.detach().numpy(), 0, 255)

    return result


def resize_image(img, size):
    size = np.floor(size)
    pixel_grid = torch.stack(torch.meshgrid(torch.arange(size[1]) / (size[1]-1)*(img.shape[1]-1), torch.arange(size[1]) / (size[1]-1)*(img.shape[1]-1)))

    return torch_interpolation(pixel_grid, torch.tensor(img)).transpose(0,1).long().numpy()


if __name__ == "__main__":
    img = array(Image.open("lena_color.gif").convert("RGB"))
    img_size = img.shape[0]
    img = img[0:img_size:4, 0:img_size:4]
    img_size = img.shape[0]

    mesh_size = 5

    delta = img_size / (mesh_size - 1.)
    mesh = np.ones((2, mesh_size + 3, mesh_size + 3))
    for i in range(mesh_size + 3):
        for j in range(mesh_size + 3):
            mesh[:, i, j] = [(i - 1) * delta, (j - 1) * delta]

    mesh_trans = mesh + 0.2 * delta / 5 * np.random.randn(np.size(mesh, 0), np.size(mesh, 1), np.size(mesh, 2))
    mesh_trans = torch.tensor(mesh_trans)

    img = torch.tensor(img)
    mesh = torch.tensor(mesh)
    mesh_no_last_row = mesh[:, 0:mesh_size + 2, 0:mesh_size + 2]
    mesh_trans_no_last_row = mesh_trans[:, 0:mesh_size + 2, 0:mesh_size + 2]

    pixel_grid = torch.stack(torch.meshgrid(torch.arange(img_size), torch.arange(img_size)))


    diff = torch_transformation(pixel_grid, mesh_trans, delta)

    plt.scatter(diff[0],diff[1])
    plt.show()

    result = (128 * torch.ones_like(img)).float().requires_grad_(True)
    opt = torch.optim.Adam([result], lr=40)
    st = time.time()
    for iter in range(5):
        img_cycled = reverse_mapping(result.clamp(0, 255), mesh_trans, delta)
        ssd = ((img_cycled - img) ** 2).mean()
        opt.zero_grad()
        ssd.backward()
        opt.step()
        print("iter:", iter, "loss:", ssd.item())
    print(time.time() - st)
    img_cycled_np = img.detach().long().numpy()
    result_np = result.detach().long().numpy()
    imshow(img_cycled_np)
    plot(mesh_no_last_row[0], mesh_no_last_row[1], 'orange')
    plot(mesh_no_last_row.T[..., 0], mesh_no_last_row.T[..., 1], 'orange')
    show()
    imshow(np.clip(result_np, 0, 255), cmap="gray")
    plot(mesh_trans_no_last_row[0], mesh_trans_no_last_row[1], 'orange')
    plot(mesh_trans_no_last_row.T[..., 0], mesh_trans_no_last_row.T[..., 1], 'orange')
    show()

    # # another kind of optimization method for computing warped image
    # disp = torch.zeros_like(current_mesh).requires_grad_(True)
    # opt = torch.optim.Adam([disp], lr=2e-1)
    # st = time.time()
    # for iter in range(50):
    #     mesh_trans_inv = original_mesh + disp
    #     img_cycled = reverse_mapping(img_trans, mesh_trans_inv, mesh_spacing)
    #     ssd = ((img_cycled - img) ** 2).mean()
    #     opt.zero_grad()
    #     ssd.backward()
    #     opt.step()
    #     # print("iter:", iter, "loss:", ssd.item())
    #     # if iter % 50 == 0:
    #     #     imshow(img_cycled.detach().numpy(), cmap="gray")
    #     #     show()
    # print(time.time()-st)
    # print("iter:", iter, "loss:", ssd.item())
    # imshow(img_cycled.detach().numpy(), cmap="gray")
    # show()
