
# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt


# In[48]:

N=18000
times=(np.arange(0,N))
R = np.zeros(N)
R[1] = 400
F = np.zeros(N)
F[1] = 200
k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
dt = 0.05


# In[49]:

#for i in range(1,8000-1):
for i in range(1,len(R)-1):
    R[i+1], F[i+1] = R[i] + dt *(k1 * R[i] - k2 * R[i] * F[i]), F[i] + dt *(k3 * R[i] * F[i] - k4 * F[i])


# In[50]:

plt.plot(R)
plt.plot(F)
plt.show()


# In[65]:

N=18000
dt = 0.05
t=(np.arange(0,N))*dt
k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04

from scipy.integrate import odeint
def pend(y,t):
    R,F = y
    dydt=[k1*R-k2*R*F,k3*R*F-k4*F]
    return dydt

y0=[400. , 200.]

sol = odeint(pend, y0, t)
plt.plot(t, sol[:, 0], 'b', label='Rabbits')
plt.plot(t, sol[:, 1], 'g', label='Foxes')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


# In[57]:

get_ipython().magic('pinfo odeint')


# In[43]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt
import random
#random.seed(1) 
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
end_time = 600

def get_rates(rabbits, foxes):
    """
    Return the rates (expected events per day) as a tuple:
    (rabbit_birth, rabbit_death, fox_birth, fox_death)
    """
    rabbit_birth = k1 * rabbits 
    rabbit_death = k2 * rabbits * foxes
    fox_birth = k3 * rabbits * foxes 
    fox_death = k4 * foxes
    return (rabbit_birth, rabbit_death, fox_birth, fox_death)

dead_foxes = 0
dead_everything = 0
runs = 100

second_peak_times = []
second_peak_foxes = []

mean_times = np.zeros(runs)
mean_foxes = np.zeros(runs)
upper_quartile_times = np.zeros(runs)
lower_quartile_times = np.zeros(runs)
upper_quartile_foxes = np.zeros(runs)
lower_quartile_foxes = np.zeros(runs)


for run in range(runs):
    time = 0
    rabbit = 400
    fox = 200
    # we don't know how long these will be so start as lists and convert to arrays later
    times = []
    rabbits = []
    foxes = []

    while time < end_time:
        times.append(time)
        rabbits.append(rabbit)
        foxes.append(fox)
        (rabbit_birth, rabbit_death, fox_birth, fox_death) = rates = get_rates(rabbit, fox)
        sum_rates = sum(rates)
        if sum_rates == 0:
            # print("everything dead at t=",time)
            dead_everything += 1
            times.append(end_time)
            rabbits.append(rabbit)
            foxes.append(fox)
            break
        wait_time = random.expovariate( sum_rates )
        time += wait_time
        choice = random.uniform(0, sum_rates)
        # Imagine we threw a dart at a number line with span (0, sum_rates) and it hit at "choice"
        # Foxes change more often than rabbits, so we'll be faster if we check them first!
        choice -= fox_birth
        if choice < 0:
            fox += 1 # fox born
            continue
        choice -= fox_death
        if choice < 0:
            fox -= 1 # fox died
            if fox == 0:
                #print("Foxes all died at t=",time)
                dead_foxes += 1
                ## Break here to speed things up (and not track the growing rabbit population)
            continue
        if choice < rabbit_birth:
            rabbit += 1 # rabbit born
            continue
        rabbit -= 1 # rabbit died
    
    times = np.array(times)
    rabbits = np.array(rabbits)
    foxes = np.array(foxes)
    
    index_of_second_peak = np.argmax(foxes*(times>200)*(foxes>100))
    if index_of_second_peak:
        second_peak_times.append(times[index_of_second_peak])
        second_peak_foxes.append(foxes[index_of_second_peak])
    
    if len(second_peak_times)>0:
        mean_times[run] = np.mean(second_peak_times)
        mean_foxes[run] = np.mean(second_peak_foxes)
        upper_quartile_times[run] = np.percentile(second_peak_times,75)
        lower_quartile_times[run] = np.percentile(second_peak_times,25)
        upper_quartile_foxes[run] = np.percentile(second_peak_foxes,75)
        lower_quartile_foxes[run] = np.percentile(second_peak_foxes,25)

    # We don't want to plot too many lines, but would be fun to see a few
    if run < 50 and rank == 0:
        plt.plot(times, rabbits, 'b')
        plt.plot(times, foxes, 'g')
        
if rank !=0:
    comm.Send(mean_times[-1] , dest=0, tag=111)
    comm.Send(mean_foxes[-1] , dest=0, tag=222)
    comm.Send(runs , dest=0, tag=333)       
    comm.Send(dead_everything , dest=0, tag=444)  
    comm.Send(dead_foxes , dest=0, tag=555)    
        
        
else:
    
    total_mean_times = mean_times[-1]
    total_mean_foxes = mean_foxes[-1]
    total_runs = runs
    total_dead_everything = dead_everything
    total_dead_foxes = dead_foxes
    
    for i in range(1, size):
        comm.Recv(mean_times, ANY_SOURCE , tag=111)
        total_mean_times += mean_times
        comm.Recv(mean_foxes, ANY_SOURCE , tag=222)
        total_mean_foxes += mean_foxes       
        comm.Recv(runs, ANY_SOURCE , tag=333)
        total_runs += runs 
        comm.Recv(dead_everything, ANY_SOURCE , tag=444)
        total_dead_everything += dead_everything         
        comm.Recv(dead_foxes, ANY_SOURCE , tag=555)
        total_dead_foxes += dead_foxes         
        
        
    plt.legend(['rabbits','foxes'],loc="best") # put the legend at the best location to avoid overlapping things
    plt.ylim(0,3000)
    plt.show()

    print("Number of total runs {}".format(total_runs))
    print("Everything died {} times out of {} or {:.1f}%".format(dead_everything, runs, 100*dead_everything/runs))
    print("Foxes died {} times out of {} or {:.1f}%".format(dead_foxes, runs, 100*dead_foxes/runs))

    plt.semilogx(mean_times,'-r')
    plt.semilogx(upper_quartile_times,':r')
    plt.semilogx(lower_quartile_times,':r')
    plt.ylabel('Second peak time (days)')
    plt.xlim(10)
    plt.show()
    print("Second peak (days) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(mean_times[-1], lower_quartile_times[-1], upper_quartile_times[-1]))


    plt.semilogx(mean_foxes,'-k')
    plt.semilogx(upper_quartile_foxes,':k')
    plt.semilogx(lower_quartile_foxes,':k')
    plt.ylabel('Second peak foxes')
    plt.xlim(10)
    plt.show()
    print("Second peak (foxes) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(mean_foxes[-1], lower_quartile_foxes[-1], upper_quartile_foxes[-1]))



# In[ ]:



