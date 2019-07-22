#@Autor: Lorenzi Flavio

#The entire code is taken from Spinningup dataset (tutorial).
#The code has been tested and modified in according to my aims.
#for example resize, refactoring e deleting operations, 
#to delete useful parts like tests, prints, ...

####NB: in italian I have written my personal observations, in english there are the predefined ones.


import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.

    """
    
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)  #np.zeros crea un'array di obs_dim el di taglia size ciascuno 
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)  
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)  
        
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs  
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):                      
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],                   #dict crea un dizionario
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

Soft Actor-Critic


"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,        #Valori di default per quelli modificati: 
        steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,            #steps_per_epoch=5000, epochs=50
        polyak=0.995, lr=1e-4, alpha=0.2, batch_size=100, start_steps=10000,            #start_steps=1000, max_ep_len=1000
        max_ep_len=2000, logger_kwargs=dict(), save_freq=1):                         
                                                                                    
    """


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given      
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to       
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph``obs, and actions in          2 soft Q-funtions
                                           | ``a_ph``acts.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and     
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states        State value function V
                                           | in ``x_ph``obs. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak AVERAGING for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD, for each gradient update.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout. (quante coppie azioni stato al massimo)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)       #inizializza la funzione di log
    logger.save_config(locals())

    tf.set_random_seed(seed)   #genera un seed casuale se il seed non è impostato #NB io uso seed=0 quindi questa riga è inutile! 
    np.random.seed(seed)

    #NB: env, obs, action

    env = env_fn()

    
    obs_dim = env.observation_space.shape[0]  #prendo il primo elemento del vettore shape di obs
    

    act_dim = env.action_space.shape[0]

    # upper bound per le azioni
    act_limit = env.action_space.high[0]

    # condivisione di info attraverso keyword
    ac_kwargs['action_space'] = env.action_space

    # Input per la rete (grafo tensorflow)  
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    #NB qui vedremo la funzione Actor Critic che ha l'obiettivo di generare i vari 
    #output della rete, ricevendo come input coppie azione stato (ph ---> placeholder)
    #NB --> actor decide le azioni, critic le valuta

    # Output 
    with tf.variable_scope('main'):     #funzione tf che assicura un buon managing dei valori principali della rete
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)    
    
    # Valori target della rete,  ovvero l'output desiderato! ! ! 
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    #in questo modo ho creato una sorta di range di valori generati


    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # contatore per le variabili
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)





  	#Sezione relativa alla perdita del gradiente con le rispettive formule! ! !

  	#NB: dato un input X e un output desiderato T, si genera un output reale Y: l'errore sarà T - Y

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)		#si prende la minima tra le due Q function impiegate

    # Targets for Q and V regression
    #stop gradient è una funzione tf che arresta il calcolo del gradiente e calcola i dati:
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)  #q predetta #B   
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)    #v predetta

    # Soft actor-critic losses
    # loss functions: "ci fanno capire quanto funziona la rete"
    # reduce_mean : calcola la media degli elementi (tra le dimensioni di un tensore)

    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)

    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)    	#formula teoria: (predetta - attuale)**2 
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss     






    #----- Ottimizzazione con funzione Adam per il training e Polyak per il target

    # POLICY TRAIN OPERATION
    # (has to be separate from value train op, because q1_pi appears in pi_loss)

    #Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure
    #Per aggiornare i pesi della rete iterativa in base ai dati di training.
    
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))  #NB con get_vars minimizza la perdita su tutte le variabili trainabili

    # VALUE TRAIN OPERATION
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)   
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')						
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)


    # Polyak averaging for target variables   (la rete si aggiorna grazie a questo fattore e tiene conto della MEDIA)
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    #group crea un insieme di operazioni
    #assign genera un Tensore che mantiene il valore di riferimento dopo che questo è stato assegnato (AGGIORNA VTARG A SECONDA DEL RISULTATO OTTENUTO DOPO)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)   
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    #-----





    # All operations to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    #Nuova sessione Tensorflow! ! ! ! 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving (nel logger)   --->  end operation!
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})




    #metodo che avvia la sessione per l'azione stocastica
    def get_action(o, deterministic=False):         #può essere settato a true tra gli hyperparams ----> non serve per il mio scopo!
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]
      
 


    #----step ed episodi genereati fino al done (end)-----#
    
    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


    total_steps = steps_per_epoch * epochs          #prodotto tra gli imput della rete che quindi sono proporzionali

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):                    #inizio ciclo training ! ! !

        """
        IMPORTANTE 
        Finchè start_steps non è trascorso, campiona random le azioni
        da una distribuzione uniforme per un esplorazione migliore.
        Dopodiche usa la policy imparata
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)  #genera random action
        ep_ret += r
        ep_len += 1             #incrementa ep_len fino a max_ep_len   dopodichè   done

        # Ignore the "done" signal if it comes from hitting the time horizon
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        #update most recent obs
        o = o2



        if d or (ep_len == max_ep_len):
            """
            Esegui tutti gli update SAC alla fine della traiettoria

            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)      #usa il dizionario batch contenente i campioni
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }										
                outs = sess.run(step_ops, feed_dict)                        #formazione di una coppia azione-stato
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)					#Chiudi sessione con questi valori
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up (crea intervalli di salvataggio fino alla fine)
        if t > 0 and t % steps_per_epoch == 0:      # quando i total steps sono divisibili per spe  : salva la divisione dentro epoch
            epoch = t // steps_per_epoch			# checkpoint

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)                   #Salva il nuovo stato

           
            # Log info about epoch ( IMMAGAZZINA & STAMPA )

            logger.log_tabular('Epoch', epoch)							#log_tabular associa ogni chiave al suo valore, di cui è specificato un parametro (per esempio =True)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)            #somma max di tutti gli episodi
            logger.log_tabular('Q1Vals', with_min_and_max=True) 	
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


    #--------------------#


            #Main: hyperparams principali, con valori di default, usati se non modificati nel launcher

if __name__ == '__main__':   
    import argparse                         #Il modulo argparse semplifica la scrittura di interfacce da riga di comando.
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='Ambiente-in-questione-v0')  
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)  #lunghezza di ogni episode
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='senzaNome')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,   #lambda function trasforma gym.make in environment chiamabile nei parametri
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),                 #riduzione per le shortcut con vecchioTermine = args.nuovoTermine
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
