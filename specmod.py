from netpython import models,pynet
import numpy as np
import pylab
from scipy.stats import binned_statistic


# ------ SIMULATION FUNCTIONS -------------

def fullrun(net,p):

    '''Runs one simulation over net, with deleterious mutation probability p'''

    N=len(net)

    substituted=set() # set of nodes with substitutions, initially empty

    NI=0 # total number of incompatibilities 

    edges=list(net.edges)
    nodes=list(net)

    done=False

    while not(done):

        targetnode=np.random.choice(nodes) # pick one node at random for substitution

        if not(targetnode in substituted): # if not already substituted, add it

            substituted.add(targetnode)

        NI+=1 # number of substitutions grows by one per round (even if target node mutated already)

        for neighbour in net[targetnode]: # check all neighbours of target node

            if neighbour in substituted: # if there is a mutation, 

                if np.random.rand()<p: # it is deleterious with probability p

                    done=True
   
    return NI # number of substitutions 'til deleterious BDMI arises

def onerun(net,maxK,p=1.0,stopper=False,threshold=0):

    '''Same as above but with more details and more outputs.
       maxK = max number of potential incompatibilities before run stopped,
       threshold = number of realized incompatibilities required for speciation,
       stopper = False -> always run until maxK potential incompatibilities,
       True -> run until threshold for realized incompatibilities reached'''

    N=len(net)

    if maxK>N:

        maxK=N

    substituted=set()

    NI=0

    NI_sum=[]
    KN=[]
    RI=[]

    Nsubs=0

    nodes=list(net) # list of nodes in network net

    found=False

    incompatibilities=0

    while (Nsubs<maxK) and not(found): # runs only until maxK substitutions

        targetnode=np.random.choice(nodes)  # pick a random node

        if not(targetnode in substituted):

            substituted.add(targetnode)
            Nsubs+=1

            for neighbour in net[targetnode]: # go through target's network neighbours; now only one subs per node (?) PROBLEM otherwise: how to count NI

                if neighbour in substituted:

                    NI+=1 # number of potential incompatibilities (substitutions connected by a link) increases

                    if stopper: # if stopping rule used (stops when deleterious incompatibility encountered)

                        if np.random.rand()<p: # probability of incompatibility being harmful

                            incompatibilities+=1

                            if incompatibilities>threshold: # if enough harmful incompatibilities, they become deleterious

                                found=True

            RI.append(incompatibilities) # number of REAL incompatibilities
            NI_sum.append(NI) # number of arisen POTENTIAL incompatibilities
            KN.append(Nsubs/float(N)) # number of (successful) substitutions / network size

    return NI_sum,KN,RI

# ---------- FUNCTIONS FOR ITERATED RESULTS -----------

def times_all(N,m,rounds,p=0.01):

    '''Yields lists of times to speciation for power law and random nets
       of size N and average degree of 2m, computed over rounds iterations,
       and using probability p for a single substitution being deleterious'''
    
    powerlaws=[]

    for x in xrange(0,rounds):

        net=pynet.SymmNet()

        net=models.makeAB(N,m)

        powerlaws.append(fullrun(net,p))

        if x%20==0:

            print("Round "+str(x))

    print("power laws done")

    randomnets=[]

    for x in xrange(0,rounds):

        net=pynet.SymmNet()

        net=makequickER(N,N*m)

        randomnets.append(fullrun(net,p))

        if x%20==0:

            print("Round "+str(x))

    return powerlaws,randomnets

def run_all(N,m,maxK,rounds,p=0.01,stopper=False,threshold=0):

    '''Runs and plots results for power law and random networks
       of N nodes and an average degree of 2m, iterated over rounds runs.
       maxK = max number of incompatibilities before stopping,
       threshold = number of incompatibilities required for speciation.
       Stopper: True = run simulations until threshold reached,
       False = always run simulation until maxK incompatibilities'''

    f=pylab.figure()

    if stopper:

        f.suptitle('p='+str(p)+", N="+str(N)+", E="+str(N*m))

    else:

        f.suptitle("N="+str(N)+", E="+str(N*m))


    # ------- init vectors for data collection 

    Kseries=[]

    pltimes=[]

    for i in xrange(0,maxK):

        Kseries.append(float(i)/N)
    
    NpI_series=[]
    NrI_series=[]

    for i in range(0,maxK):

        NpI_series.append([])
        NrI_series.append([])

    # -------- first model: BA ------------------

    for r in xrange(0,rounds): # rounds = total # of iterations

        net=models.makeAB(N,m)

        NpI,K,NrI=onerun(net,maxK,p=p,stopper=stopper,threshold=threshold) # potential incompatibilities, substitutions, real incompatibilities

        pltimes.append(K[-1]) # time to speciation in substitutions = last entry in vector

        # plot NpI-series

        ax1=f.add_subplot(3,2,1)

        ax1.plot(K,NpI,'r-',alpha=0.3)

        ax2=f.add_subplot(3,2,2)
        ax2.plot(K,NrI,'r-',alpha=0.3)

        #ax.plot([x[-1],x[-1]],[0,50],'k-',alpha=0.2,zorder=1)

        for i in range(0,len(NpI)):

            NpI_series[i].append(NpI[i])
            NrI_series[i].append(NrI[i])

        if r%100==0:

            print("Round "+str(r))

    meanseries_NpI=[]
    stdseries_hi_NpI=[]
    stdseries_lo_NpI=[]
    varseries_NpI=[]

    for i in range(0,len(NpI_series)):

        meanseries_NpI.append(np.mean(NpI_series[i]))
        stdseries_hi_NpI.append(np.mean(NpI_series[i])+np.std(NpI_series[i]))
        stdseries_lo_NpI.append(np.mean(NpI_series[i])-np.std(NpI_series[i]))
        varseries_NpI.append(np.var(NpI_series[i]))

    ax1.plot(Kseries,meanseries_NpI,'k-')
    ax1.plot(Kseries,stdseries_hi_NpI,'k--')
    ax1.plot(Kseries,stdseries_lo_NpI,'k--')

    ax1.set_xlabel('K/N')
    ax1.set_ylabel('# of potential inc.')
    ax1.set_title('power-law net')
    ax1.set_xlim(0,float(maxK)/N)

    meanseries_NrI=[]
    stdseries_hi_NrI=[]
    stdseries_lo_NrI=[]
    varseries_NrI=[]

    for i in range(0,len(NrI_series)):

        meanseries_NrI.append(np.mean(NrI_series[i]))
        stdseries_hi_NrI.append(np.mean(NrI_series[i])+np.std(NrI_series[i]))
        stdseries_lo_NrI.append(np.mean(NrI_series[i])-np.std(NrI_series[i]))
        varseries_NrI.append(np.var(NrI_series[i]))

    ax2.plot(Kseries,meanseries_NrI,'k-')
    ax2.plot(Kseries,stdseries_hi_NrI,'k--')
    ax2.plot(Kseries,stdseries_lo_NrI,'k--')

    ax2.set_xlabel('K/N')
    ax2.set_ylabel('# of actual inc.')
    ax2.set_title('power-law net')
    ax2.set_xlim(0,float(maxK)/N)

    # ER HERE REPEAT

     # ------- init vectors for data collection 

    Kseries=[]

    ertimes=[]

    for i in xrange(0,maxK):

        Kseries.append(float(i)/N)
    
    NpI_series=[]
    NrI_series=[]

    for i in range(0,maxK):

        NpI_series.append([])
        NrI_series.append([])

    # -------- first model: BA ------------------

    for r in xrange(0,rounds): # rounds = total # of iterations

        net=makequickER(N,m*N)

        NpI,K,NrI=onerun(net,maxK,p=p,stopper=stopper,threshold=threshold) # pot incompatibilities, substitutions, real incompatibilities 

        ertimes.append(K[-1]) # time to speciation in substitutions = last entry in vector

        # plot NpI-series

        ax3=f.add_subplot(3,2,3)

        ax3.plot(K,NpI,'g-',alpha=0.3)

        ax4=f.add_subplot(3,2,4)
        ax4.plot(K,NrI,'g-',alpha=0.3)

        #ax.plot([x[-1],x[-1]],[0,50],'k-',alpha=0.2,zorder=1)

        for i in range(0,len(NpI)):

            NpI_series[i].append(NpI[i])
            NrI_series[i].append(NrI[i])

        if r%100==0:

            print("Round "+str(r))

    er_meanseries_NpI=[]
    er_stdseries_hi_NpI=[]
    er_stdseries_lo_NpI=[]
    er_varseries_NpI=[]

    for i in range(0,len(NpI_series)):

        er_meanseries_NpI.append(np.mean(NpI_series[i]))
        er_stdseries_hi_NpI.append(np.mean(NpI_series[i])+np.std(NpI_series[i]))
        er_stdseries_lo_NpI.append(np.mean(NpI_series[i])-np.std(NpI_series[i]))
        er_varseries_NpI.append(np.var(NpI_series[i]))

    ax3.plot(Kseries,er_meanseries_NpI,'k-')
    ax3.plot(Kseries,er_stdseries_hi_NpI,'k--')
    ax3.plot(Kseries,er_stdseries_lo_NpI,'k--')

    ax3.set_xlabel('K/N')
    ax3.set_ylabel('# of potential inc.')
    ax3.set_title('random net')
    ax3.set_xlim(0,float(maxK)/N)

    maxy=max(ax1.get_ylim()[1],ax3.get_ylim()[1])

    ax1.set_ylim(0,maxy)
    ax3.set_ylim(0,maxy)

    er_meanseries_NrI=[]
    er_stdseries_hi_NrI=[]
    er_stdseries_lo_NrI=[]
    er_varseries_NrI=[]

    for i in range(0,len(NrI_series)):

        er_meanseries_NrI.append(np.mean(NrI_series[i]))
        er_stdseries_hi_NrI.append(np.mean(NrI_series[i])+np.std(NrI_series[i]))
        er_stdseries_lo_NrI.append(np.mean(NrI_series[i])-np.std(NrI_series[i]))
        er_varseries_NrI.append(np.var(NrI_series[i]))

    ax4.plot(Kseries,er_meanseries_NrI,'k-')
    ax4.plot(Kseries,er_stdseries_hi_NrI,'k--')
    ax4.plot(Kseries,er_stdseries_lo_NrI,'k--')

    ax4.set_xlabel('K/N')
    ax4.set_ylabel('# of actual inc.')
    ax4.set_title('random net')
    ax4.set_xlim(0,float(maxK)/N)

    maxy=max(ax2.get_ylim()[1],ax4.get_ylim()[1])

    ax2.set_ylim(0,maxy)
    ax4.set_ylim(0,maxy)

    max

    relseries_NpI=[]
    relseries_NrI=[]

    maxindex=min(len(varseries_NpI),len(er_varseries_NpI))

    for i in range(0,maxindex):

        relseries_NpI.append(varseries_NpI[i]/er_varseries_NpI[i])
        relseries_NrI.append(varseries_NrI[i]/er_varseries_NrI[i])

    ax5=f.add_subplot(3,2,5)

    ax5.plot(Kseries,relseries_NpI,'m-',label='pot.')
    ax5.plot(Kseries,relseries_NrI,'b-',label='act.')

    ax5.plot([Kseries[0],Kseries[-1]],[1.0,1.0],'k--',color=[0.5,0.5,0.5])

    ax5.legend(loc=0)
    
    ax5.set_xlabel('K/N')
    ax5.set_ylabel('power law var/random var')
    ax5.set_xlim(0,float(maxK)/N)
    ax5.set_ylim(0,5.0)

    ax6=f.add_subplot(3,2,6)

    plot_BDMItimes(pltimes,ertimes,numbins=20,ax=ax6)
    ax6.set_xlim(0,float(maxK)/N)
    
    return pltimes,ertimes

# ---------- NETWORK GENERATORS ----------

def makequickER(N,E):

    '''Creates a random (Erdos-Renyi) network with N nodes and E randomly placed edges'''

    net=pynet.SymmNet()

    for i in range(0,E):

        found=False

        while not(found):

            node_i=np.random.randint(N)+1
            node_j=np.random.randint(N)+1

            if net[node_i][node_j]==0.0:

                if not(node_i==node_j):

                    net[node_i][node_j]=1.0
                    found=True

    return net

# --------- functions for plotting BDMI times in two lists (power law nets, random nets) ------

def plot_BDMItimes(powerlawtimes,randomtimes,numbins=20,ax=-9):

    if ax==-9:

        fig=pylab.figure()

        ax=fig.add_subplot(1,1,1)

    # pl first

    maxval=max(max(powerlawtimes),max(randomtimes))
    minval=min(min(powerlawtimes),min(randomtimes))
    

    bins=np.linspace(0.95*float(minval),1.05*float(maxval),numbins)

    pdata_powerlaw,bin_edges=np.histogram(powerlawtimes,bins=bins,density=True)

    pdata_random,bin_edges_random=np.histogram(randomtimes,bins=bins,density=True)

    bincenters,_,_=binned_statistic(powerlawtimes,powerlawtimes,statistic="mean",bins=bins)

    bincenters_random,_,_=binned_statistic(randomtimes,randomtimes,statistic='mean',bins=bins)

    zerolist=[]

    for b in bincenters:

        zerolist.append(0)

    ax.fill_between(bincenters,pdata_powerlaw,zerolist,color='r',alpha=0.3,zorder=0)

    ax.fill_between(bincenters_random,pdata_random,zerolist,color='b',alpha=0.3)

def plot_cumtimes(pl,rn,ax=-9,title=''):

    if ax==-9:

        fig=pylab.figure()
        ax=fig.add_subplot(1,1,1)
        fig.suptitle(title)

    pl_sorted=sorted(pl)
    rn_sorted=sorted(rn)

    pl_y=[]
    rn_y=[]

    for i in range(0,len(pl_sorted)):

        pl_y.append(float(i)/len(pl_sorted))

    for i in range(0,len(rn_sorted)):
        
        rn_y.append(float(i)/len(rn_sorted))

    ax.plot(pl_sorted,pl_y,'r-')
    ax.plot(rn_sorted,rn_y,'b-')

# ------- AUXILIARY FUNCTIONS AND LEFTOVERS -------------

def p_from_list(data):

    xvalues_rough=sorted(data)

    xlist=[]

    ylist=[]

    for i in range(0,len(xvalues_rough)-2):

        if xvalues_rough[i+1]>xvalues_rough[i]:

            xlist.append(xvalues_rough[i])

            ylist.append(i/float(len(xvalues_rough)))

    return xlist,ylist


    

                

    

    

                                

    

        

        

        

    

                    

                

        

        
