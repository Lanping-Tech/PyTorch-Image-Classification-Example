import os
import matplotlib.pyplot as plt

def performance_display(metric_value, metric_name, output_path):
    color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for index, (name, values) in enumerate(metric_value.items()):
        plt.plot(list(range(1, len(values)+1)), 
                    values, 
                    color=color[index], 
                    linewidth=1.5, 
                    label=name)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    # plt.xticks(list(range(1, len(values)+1)),list(range(1, len(values)+1)))
    plt.grid(linestyle='--')
    fig_path = os.path.join(output_path, metric_name+'.png')
    plt.savefig(fig_path, dpi=500, bbox_inches = 'tight')
    plt.cla()