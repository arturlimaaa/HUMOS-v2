;; World2Data PDDL Domain
;; Action-Centric Ground Truth for Humanoid Navigation
;;
;; Defines kitchen action types that a robot task planner can consume.
;; Generated action traces reference these action schemas.

(define (domain kitchen-actions)
  (:requirements :strips :typing)

  (:types
    agent - object
    item - object
    location - object
  )

  (:predicates
    (at ?a - agent ?l - location)
    (holding ?a - agent ?o - item)
    (hand_free ?a - agent)
    (on ?o - item ?l - location)
    (inside ?o - item ?c - item)
    (reachable ?a - agent ?o - item)
    (is_open ?o - item)
    (is_closed ?o - item)
    (is_on ?o - item)
    (is_off ?o - item)
    (is_clean ?o - item)
    (is_cut ?o - item)
    (is_peeled ?o - item)
    (is_mixed ?o - item)
  )

  (:action open_object
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o) (is_closed ?o) (hand_free ?a))
    :effect (and (is_open ?o) (not (is_closed ?o)))
  )

  (:action close_object
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o) (is_open ?o) (hand_free ?a))
    :effect (and (is_closed ?o) (not (is_open ?o)))
  )

  (:action take_object
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o) (hand_free ?a))
    :effect (and (holding ?a ?o) (not (hand_free ?a)))
  )

  (:action put_object
    :parameters (?a - agent ?o - item ?l - location)
    :precondition (holding ?a ?o)
    :effect (and (on ?o ?l) (hand_free ?a) (not (holding ?a ?o)))
  )

  (:action turn_on
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o) (is_off ?o))
    :effect (and (is_on ?o) (not (is_off ?o)))
  )

  (:action turn_off
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o) (is_on ?o))
    :effect (and (is_off ?o) (not (is_on ?o)))
  )

  (:action wash_object
    :parameters (?a - agent ?o - item)
    :precondition (and (at ?a sink_area) (holding ?a ?o))
    :effect (is_clean ?o)
  )

  (:action cut_object
    :parameters (?a - agent ?o - item)
    :precondition (and (holding ?a knife) (reachable ?a ?o))
    :effect (is_cut ?o)
  )
)
