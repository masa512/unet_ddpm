import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_image_animation(img_seq,const_img,save_path = ".", title = " " ,reverse = False):
    """
    Creates and saves animation from plots

    ---Input---
    img_seq : list of np.array(dx,dy) for image data
    const_img : constant image used for reference
    save_path : (str) folder where to save the animation
    title : (str) Title to the plot
    reverse : (bool) whether to traverse backwards

    """

    # Parameters
    n_frames = len(img_seq)

    # Define the f,ax and blank Image plot obj
    f,axes = plt.subplots(1,2,figsize = (10,5))
    im0 = axes[0].imshow(img_seq[0].squeeze(), cmap='gray')  # Invisible blank image  
    im1 = axes[1].imshow(const_img.squeeze(), cmap='gray')  # Invisible blank image  
    # Define init callable (base case when animation begins)
    def init():
        for ax in axes:
          ax.set_xlim(-3, 3)
          ax.set_ylim(-3, 3)
        axes[0].set_title(title)
        axes[1].set_title('Reference')
        return (im0,im1)
      
    # Define update callable for iterative animation building
    def update(step):
        # Stepping based on whether to reverse traversal or not
        i = step
        if reverse:
          i = (n_frames - 1) - step

        # Update the new frame
        im0.set_array(img_seq[i].squeeze())
        axes[0].set_title(f"{title} - {i+1}/{n_frames}")
        return (im0,im1)

    # Create animation object
    animation = FuncAnimation(f, update, frames= n_frames, blit=True)

    # Export animation and close the figure
    animation.save(save_path, writer="Pillow", fps=20)
    plt.close(f)
    print("Finish saving gif file: ", save_path)