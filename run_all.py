import subprocess
import csv
from run_all_config import args

FILENAME = "nmi_scores.csv"

with open(FILENAME, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Method", "NMI Score"])

    # Creating the Stochastic Block Model network
    # The network will be stored in an edgelist file
    subprocess.run(["python", "sbm_simulator.py", "-p1", str(args.probability1),
                                                  "-p2", str(args.probability2),
                                                  "-p3", str(args.probability3)], 
                    check=True)


    # Learning the SBM's node embeddings using spectral clustering and running kmeans
    # on them and extracting the NMI score
    print("Running Spectral Clustering...")
    result_spectral = subprocess.run(["python", "SpectralClusteringImplementation/laplacian_eigenmap.py"], 
                                     capture_output=True, text=True, check=True)
    output_spectral = result_spectral.stdout.strip()
    nmi_spectral = output_spectral.split("NMI Score:")[-1].strip() # obtaining the NMI score

    writer.writerow(["Spectral Clustering", nmi_spectral])


    # Learning the node embeddings using the node2vec method
    print("Running node2vec...")
    subprocess.run(["python", "node2vecImplementation/n2v_embedding_gen.py"], check=True)

    # Running kmeans on the embeddings and extracting the NMI score
    result_n2v = subprocess.run(["python", "node2vecImplementation/n2v_kmeans_eval.py"], 
                                capture_output=True, text=True, check=True)
    output_n2v = result_n2v.stdout.strip()
    nmi_n2v = output_n2v.split("NMI Score:")[-1].strip() # obtaining the NMI score
    writer.writerow(["Node2vec", nmi_n2v])


    # Learning the node embeddings using LINE method
    print("Running LINE...") 
    line_subprocess = subprocess.Popen(["python", "LINEImplementation/train.py", "-g", 
                      "./sbm_graph.edgelist", "-save", "LINEImplementation/LINE_model.pt", 
                      "-lossdata", "LINEImplementation/loss_data.pkl", 
                      "-epochs", "10", "-batchsize", "512", "-dim", "64"])
    line_subprocess.wait()
    
    # Running kmeans on the embeddings and extracting the NMI score
    result_line = subprocess.run(["python", "LINEImplementation/kmeans.py"], 
                                 capture_output=True, text=True, check=True)
    output_line = result_line.stdout.strip()
    nmi_line = output_line.split("NMI Score:")[-1].strip() # obtaining the NMI score

    writer.writerow(["LINE", nmi_line])


    # Learning the node embeddings using GraRep method
    print("Running GraRep...")
    grarep_subprocess = subprocess.Popen(["python", "GraRepImplementation/main.py", "-g", "./sbm_graph.edgelist", 
                      "-order", "6", "-dim", "21", "-iters", "20"])
    grarep_subprocess.wait()
    
    # Running kmeans on the embeddings and extracting the NMI score
    result_grarep = subprocess.run(["python", "GraRepImplementation/kmeans.py"], 
                                   capture_output=True, text=True, check=True)
    output_grarep = result_grarep.stdout.strip()
    nmi_grarep = output_grarep.split("NMI Score:")[-1].strip() # obtaining the NMI score

    writer.writerow(["GraRep", nmi_grarep])

print(f"The NMI scores for each of the methods have been written to {FILENAME}")